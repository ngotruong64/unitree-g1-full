"""Play a checkpoint with a cyclic push schedule:
  Phase 0 (10s): no push
  Phase 1 (10s): push 100 N
  Phase 2 (10s): push 200 N
  Phase 3 (10s): push 300 N
  → repeat
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play with cyclic push schedule.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--real_time", action="store_true", default=False)
parser.add_argument("--push_body", type=str, default="torso_link")
parser.add_argument("--push_interval", type=float, default=2.0, help="Seconds between pushes within an active phase.")
parser.add_argument("--push_duration", type=float, default=0.2, help="Seconds each push is held.")
parser.add_argument("--phase_duration", type=float, default=10.0, help="Duration of each phase in seconds.")
parser.add_argument("--push_forces", type=float, nargs="+", default=[0, 100, 200, 300],
                    help="Force (N) per phase. Phase 0 is always rest. Default: 0 100 200 300.")
parser.add_argument("--push_seed", type=int, default=None)
parser.add_argument(
    "--phase_vel_x", type=float, nargs="+", default=None,
    help="Forward velocity (m/s) to command each phase. Same count as --push_forces. "
         "E.g. --phase_vel_x 0.8 0.5 0.5 0.5  → fast walk in phase 0, normal in rest.",
)
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

PHASE_COLORS = ["\033[92m", "\033[93m", "\033[33m", "\033[91m"]  # green, yellow, orange, red
RESET = "\033[0m"


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

    # export
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

    # === setup ===
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

    dt = env.unwrapped.step_dt
    phase_forces = args_cli.push_forces
    num_phases = len(phase_forces)
    steps_per_phase = max(1, int(round(args_cli.phase_duration / dt)))
    steps_between_pushes = max(1, int(round(args_cli.push_interval / dt)))
    steps_per_push = max(1, int(round(args_cli.push_duration / dt)))

    zeros_force = torch.zeros(num_envs, len(body_ids), 3, device=device)
    zeros_torque = torch.zeros(num_envs, len(body_ids), 3, device=device)

    # velocity override per phase (None = keep env default sampling)
    phase_vel_x = args_cli.phase_vel_x
    if phase_vel_x is not None and len(phase_vel_x) != num_phases:
        raise ValueError(f"--phase_vel_x must have {num_phases} values (one per phase), got {len(phase_vel_x)}")

    cmd_buf = env.unwrapped.command_manager._terms["base_velocity"].command  # shape (num_envs, 3): vx, vy, wz

    print(f"[INFO] Cycle: {num_phases} phases × {args_cli.phase_duration}s each")
    for i, f in enumerate(phase_forces):
        color = PHASE_COLORS[i % len(PHASE_COLORS)]
        label = "REST (no push)" if f == 0 else f"PUSH  {f:.0f} N"
        vel_note = f", vx={phase_vel_x[i]:.2f} m/s" if phase_vel_x else ""
        print(f"  Phase {i}: {color}{label}{vel_note}{RESET}")

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    step_idx = 0
    push_remaining = 0
    current_force = zeros_force.clone()
    current_phase = -1
    push_count = 0

    while simulation_app.is_running():
        start_time = time.time()

        # --- phase logic ---
        phase = (step_idx // steps_per_phase) % num_phases
        force_magnitude = phase_forces[phase]

        # announce phase change and override velocity command
        if phase != current_phase:
            current_phase = phase
            color = PHASE_COLORS[phase % len(PHASE_COLORS)]
            elapsed_s = step_idx * dt
            vel_note = ""
            if phase_vel_x is not None:
                cmd_buf[:, 0] = phase_vel_x[phase]
                vel_note = f", vx={phase_vel_x[phase]:.2f} m/s"
            if force_magnitude == 0:
                print(f"\n{color}[t={elapsed_s:.1f}s] Phase {phase}: REST — no push{vel_note}{RESET}")
            else:
                print(f"\n{color}[t={elapsed_s:.1f}s] Phase {phase}: PUSH {force_magnitude:.0f} N{vel_note}{RESET}")

        # --- push trigger ---
        step_in_phase = step_idx % steps_per_phase
        if force_magnitude > 0 and step_in_phase % steps_between_pushes == 0:
            f = torch.rand(num_envs, len(body_ids), 3, generator=rng, device=device) * 2.0 - 1.0
            # only horizontal (x, y)
            f[:, :, 2] = 0.0
            f = f / (f.norm(dim=-1, keepdim=True).clamp(min=1e-6)) * force_magnitude
            current_force = f
            push_remaining = steps_per_push
            push_count += 1
            print(f"  [PUSH {push_count}] step={step_idx}, |F|={force_magnitude:.0f} N")

        # --- apply force ---
        if push_remaining > 0 and force_magnitude > 0:
            robot.set_external_force_and_torque(current_force, zeros_torque, body_ids=body_ids)
            push_remaining -= 1
        else:
            robot.set_external_force_and_torque(zeros_force, zeros_torque, body_ids=body_ids)

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
