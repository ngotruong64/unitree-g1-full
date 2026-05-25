# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint and randomly push the robot to test robustness."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play an RSL-RL checkpoint with random push perturbations.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--push_body", type=str, default="torso_link", help="Body to apply pushes to.")
parser.add_argument(
    "--push_force",
    type=float,
    default=200.0,
    help="Max magnitude of random force per axis in Newtons (uniform in [-F, +F]).",
)
parser.add_argument(
    "--push_torque",
    type=float,
    default=0.0,
    help="Max magnitude of random torque per axis in Nm (uniform in [-T, +T]).",
)
parser.add_argument(
    "--push_interval", type=float, default=3.0, help="Seconds between push impulses."
)
parser.add_argument(
    "--push_duration",
    type=float,
    default=0.2,
    help="Seconds each push is held before releasing (impulse-like when small).",
)
parser.add_argument(
    "--push_axes",
    type=str,
    default="xy",
    help="Which force axes to perturb. Subset of 'xyz' (default: xy = horizontal push).",
)
parser.add_argument("--push_seed", type=int, default=None, help="RNG seed for push sampling.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import importlib.metadata as _metadata
import os
import time
import torch
from packaging.version import parse as _parse_version


def _set_wrench(robot, forces, torques, body_ids, device):
    """Apply wrench using the non-deprecated WrenchComposer API."""
    body_ids_t = torch.tensor(body_ids, dtype=torch.int32, device=device)
    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=body_ids_t,
    )

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _resolve_axis_mask(axes: str, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(3, device=device)
    for c in axes.lower():
        if c == "x":
            mask[0] = 1.0
        elif c == "y":
            mask[1] = 1.0
        elif c == "z":
            mask[2] = 1.0
    if mask.sum() == 0:
        raise ValueError(f"--push_axes must contain at least one of x/y/z, got {axes!r}")
    return mask


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
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export (so the exported/ dir is refreshed alongside this run)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if _parse_version(_metadata.version("rsl-rl-lib")) >= _parse_version("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        try:
            policy_nn = runner.alg.policy  # type: ignore[attr-defined]
        except AttributeError:
            policy_nn = runner.alg.actor_critic  # type: ignore[attr-defined]
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None) or getattr(
            policy_nn, "student_obs_normalizer", None
        )
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # === push setup ===
    scene = env.unwrapped.scene
    robot = scene["robot"]
    device = env.unwrapped.device
    num_envs = scene.num_envs

    body_ids, body_names = robot.find_bodies(args_cli.push_body)
    if len(body_ids) == 0:
        raise ValueError(f"No body matches --push_body={args_cli.push_body!r}")
    print(f"[INFO] Pushing body indices {body_ids} ({body_names})")

    rng = torch.Generator(device=device)
    if args_cli.push_seed is not None:
        rng.manual_seed(args_cli.push_seed)

    axis_mask = _resolve_axis_mask(args_cli.push_axes, device)
    torque_mask = torch.ones(3, device=device)  # full 3-axis torque when enabled

    zeros_force = torch.zeros(num_envs, len(body_ids), 3, device=device)
    zeros_torque = torch.zeros(num_envs, len(body_ids), 3, device=device)

    dt = env.unwrapped.step_dt
    steps_between_pushes = max(1, int(round(args_cli.push_interval / dt)))
    steps_per_push = max(1, int(round(args_cli.push_duration / dt)))
    step_idx = 0
    push_remaining_steps = 0
    current_force = zeros_force.clone()
    current_torque = zeros_torque.clone()
    push_count = 0

    obs = env.get_observations()
    # rsl-rl 2.3 returned (obs, extras)
    if isinstance(obs, tuple):
        obs = obs[0]

    print(
        f"[INFO] Push: force={args_cli.push_force} N on axes '{args_cli.push_axes}', "
        f"torque={args_cli.push_torque} Nm, every {args_cli.push_interval}s "
        f"({steps_between_pushes} steps), held {args_cli.push_duration}s ({steps_per_push} steps)"
    )

    while simulation_app.is_running():
        start_time = time.time()

        # trigger a new push
        if step_idx % steps_between_pushes == 0:
            f = (torch.rand(num_envs, len(body_ids), 3, generator=rng, device=device) * 2.0 - 1.0)
            current_force = f * axis_mask * args_cli.push_force
            if args_cli.push_torque > 0.0:
                t = (torch.rand(num_envs, len(body_ids), 3, generator=rng, device=device) * 2.0 - 1.0)
                current_torque = t * torque_mask * args_cli.push_torque
            else:
                current_torque = zeros_torque
            push_remaining_steps = steps_per_push
            push_count += 1
            mag = current_force.norm(dim=-1).mean().item()
            print(f"[PUSH {push_count}] step={step_idx}, mean |F|={mag:.1f} N")

        # apply (or clear) external wrench
        if push_remaining_steps > 0:
            _set_wrench(robot, current_force, current_torque, body_ids, device)
            push_remaining_steps -= 1
        else:
            _set_wrench(robot, zeros_force, zeros_torque, body_ids, device)

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        step_idx += 1

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
