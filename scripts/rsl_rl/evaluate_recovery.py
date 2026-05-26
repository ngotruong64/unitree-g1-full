# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to quantitatively evaluate push-recovery metrics (recovery time, fall rate, max deviations)."""

import argparse
import os
import sys
import time
import torch
import numpy as np
import gymnasium as gym

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate G1 push-recovery time and success rate.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments to evaluate.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity-Sim2Real", help="Name of the task.")
parser.add_argument("--push_body", type=str, default="torso_link", help="Body to apply pushes to.")
parser.add_argument(
    "--push_force",
    type=float,
    default=260.0,
    help="Force magnitude (N) applied to the torso.",
)
parser.add_argument(
    "--push_duration",
    type=float,
    default=0.2,
    help="Duration of the push in seconds.",
)
parser.add_argument(
    "--cmd_vel_x",
    type=float,
    default=0.5,
    help="Target forward velocity command (m/s) during evaluation.",
)
parser.add_argument(
    "--vel_threshold",
    type=float,
    default=0.15,
    help="Velocity error threshold (m/s) to define recovery.",
)
parser.add_argument(
    "--tilt_threshold",
    type=float,
    default=5.0,
    help="Tilt angle threshold (degrees) to define recovery.",
)
parser.add_argument(
    "--stable_steps",
    type=int,
    default=10,
    help="Number of consecutive stable steps (50Hz) required to confirm recovery.",
)
parser.add_argument(
    "--pre_push_steps",
    type=int,
    default=150,
    help="Steps to walk stably before applying the push.",
)
parser.add_argument(
    "--eval_steps",
    type=int,
    default=250,
    help="Max steps to track recovery after the push ends.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless mode to speed up evaluation unless requested otherwise
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as _metadata
from packaging.version import parse as _parse_version

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _set_wrench(robot, forces, torques, body_ids, device):
    """Apply wrench to the robot body links."""
    body_ids_t = torch.tensor(body_ids, dtype=torch.int32, device=device)
    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=body_ids_t,
    )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Setup handles
    scene = env.unwrapped.scene
    robot = scene["robot"]
    device = env.unwrapped.device
    num_envs = scene.num_envs
    dt = env.unwrapped.step_dt

    body_ids, body_names = robot.find_bodies(args_cli.push_body)
    if len(body_ids) == 0:
        raise ValueError(f"No body matches --push_body={args_cli.push_body!r}")

    # Override command buffer to a fixed evaluation velocity
    cmd_buf = env.unwrapped.command_manager._terms["base_velocity"].command
    cmd_buf[:, 0] = args_cli.cmd_vel_x
    cmd_buf[:, 1] = 0.0
    cmd_buf[:, 2] = 0.0

    print(f"\n==================================================")
    print(f" EVALUATING PUSH-RECOVERY METRICS")
    print(f"==================================================")
    print(f" * Task: {args_cli.task}")
    print(f" * Environments: {num_envs}")
    print(f" * Commanded Velocity (vx): {args_cli.cmd_vel_x:.2f} m/s")
    print(f" * Push Force: {args_cli.push_force} N (Random Horizontal Dir)")
    print(f" * Push Duration: {args_cli.push_duration} s")
    print(f" * Target threshold: Vel error < {args_cli.vel_threshold} m/s & Tilt < {args_cli.tilt_threshold} deg")
    print(f"==================================================\n")

    # Reset environment
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    # Pre-allocate metrics
    started_push_step = args_cli.pre_push_steps
    steps_per_push = max(1, int(round(args_cli.push_duration / dt)))
    ended_push_step = started_push_step + steps_per_push
    max_eval_steps = ended_push_step + args_cli.eval_steps

    zeros_force = torch.zeros(num_envs, len(body_ids), 3, device=device)
    zeros_torque = torch.zeros(num_envs, len(body_ids), 3, device=device)
    push_forces = zeros_force.clone()

    # Recovery tracking state
    consecutive_stable_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
    recovered = torch.zeros(num_envs, dtype=torch.bool, device=device)
    recovery_step = torch.zeros(num_envs, dtype=torch.int32, device=device) - 1
    failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    max_vel_error = torch.zeros(num_envs, device=device)
    max_tilt_deg = torch.zeros(num_envs, device=device)

    # Horizontal random push directions
    rng = torch.Generator(device=device)
    rng.manual_seed(42)  # Fixed evaluation seed for reproducibility
    f_dir = torch.rand(num_envs, len(body_ids), 3, generator=rng, device=device) * 2.0 - 1.0
    f_dir[:, :, 2] = 0.0  # horizontal push only
    f_dir = f_dir / (f_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6))
    push_forces = f_dir * args_cli.push_force

    step_idx = 0
    while step_idx < max_eval_steps and simulation_app.is_running():
        # Override velocity commands to keep command fixed during evaluation
        cmd_buf[:, 0] = args_cli.cmd_vel_x
        cmd_buf[:, 1] = 0.0
        cmd_buf[:, 2] = 0.0

        # Apply push force in the designated window
        if started_push_step <= step_idx < ended_push_step:
            _set_wrench(robot, push_forces, zeros_torque, body_ids, device)
        else:
            _set_wrench(robot, zeros_force, zeros_torque, body_ids, device)

        # Policy step
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # handle potential tuple return
            if isinstance(obs, tuple):
                obs = obs[0]

        # Convert dones to boolean mask
        dones_bool = dones.to(torch.bool)

        # Track metrics after push begins
        if step_idx >= started_push_step:
            # 1. Base velocity error (magnitude of difference in horizontal velocity)
            base_lin_vel_b = robot.data.root_lin_vel_b[:, :3]
            vel_err = torch.norm(base_lin_vel_b[:, :2] - cmd_buf[:, :2], dim=-1)
            max_vel_error = torch.max(max_vel_error, vel_err)

            # 2. Tilt angle (using projected gravity horizontal component)
            proj_g_xy = robot.data.projected_gravity_b[:, :2].norm(dim=-1)
            tilt_angle_rad = torch.arcsin(torch.clamp(proj_g_xy, max=0.9999))
            tilt_angle_deg = torch.rad2deg(tilt_angle_rad)
            max_tilt_deg = torch.max(max_tilt_deg, tilt_angle_deg)

            # 3. Check for falls (low height or environment reset)
            height = robot.data.root_pos_w[:, 2]
            fell = (height < 0.40) | dones_bool
            failed = failed | fell

            # 4. Check for stability
            is_stable = (vel_err < args_cli.vel_threshold) & (tilt_angle_deg < args_cli.tilt_threshold) & (~failed)
            
            # Increment stable steps counters
            consecutive_stable_steps = torch.where(
                is_stable,
                consecutive_stable_steps + 1,
                torch.zeros_like(consecutive_stable_steps)
            )

            # Mark newly recovered environments
            newly_recovered = is_stable & (consecutive_stable_steps >= args_cli.stable_steps) & (~recovered)
            if newly_recovered.any():
                recovered[newly_recovered] = True
                # Record step index when stability was first entered
                recovery_step[newly_recovered] = step_idx - args_cli.stable_steps + 1

        step_idx += 1

    # Close simulation
    env.close()

    # Process and print metrics
    recovered_cpu = recovered.cpu().numpy()
    failed_cpu = failed.cpu().numpy()
    recovery_step_cpu = recovery_step.cpu().numpy()
    max_vel_error_cpu = max_vel_error.cpu().numpy()
    max_tilt_cpu = max_tilt_deg.cpu().numpy()

    # Success: recovered and did not fail/fall during evaluation
    success_mask = recovered_cpu & (~failed_cpu)
    num_success = int(np.sum(success_mask))
    success_rate = (num_success / num_envs) * 100.0
    fall_rate = (int(np.sum(failed_cpu)) / num_envs) * 100.0

    # Calculate recovery times (in seconds)
    # 1. From push START (step 150)
    rec_time_from_start = (recovery_step_cpu[success_mask] - started_push_step) * dt
    # 2. From push END (step 150 + steps_per_push)
    rec_time_from_end = (recovery_step_cpu[success_mask] - ended_push_step) * dt

    print(f"\n==================================================")
    print(f" EVALUATION RESULTS (N={num_envs})")
    print(f"==================================================")
    print(f" * Successful Recovery Rate: {success_rate:.1f}% ({num_success}/{num_envs})")
    print(f" * Fall / Failure Rate:      {fall_rate:.1f}%")
    print(f"--------------------------------------------------")
    
    if num_success > 0:
        print(f" * Recovery Time (from START of push):")
        print(f"   - Mean: {np.mean(rec_time_from_start):.3f} s")
        print(f"   - Std:  {np.std(rec_time_from_start):.3f} s")
        print(f"   - Min:  {np.min(rec_time_from_start):.3f} s")
        print(f"   - Max:  {np.max(rec_time_from_start):.3f} s")
        print(f"")
        print(f" * Recovery Time (from END of push):")
        print(f"   - Mean: {np.mean(rec_time_from_end):.3f} s")
        print(f"   - Std:  {np.std(rec_time_from_end):.3f} s")
        print(f"   - Min:  {np.min(rec_time_from_end):.3f} s")
        print(f"   - Max:  {np.max(rec_time_from_end):.3f} s")
    else:
        print(f" [WARNING] No environments successfully recovered.")

    print(f"--------------------------------------------------")
    print(f" * Maximum Body Deviations (All Envs):")
    print(f"   - Max Vel Error: {np.max(max_vel_error_cpu):.3f} m/s")
    print(f"   - Max Tilt Angle: {np.max(max_tilt_cpu):.1f}°")
    print(f"==================================================\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
