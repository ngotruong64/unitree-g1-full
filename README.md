# Unitree G1 Humanoid — Push-Recovery Locomotion

PPO-based walking policy for the Unitree G1 (29-DoF) with sim-to-real domain randomization and external-push robustness evaluation.

---

## Stack

| Component | Choice |
|---|---|
| Simulator | Isaac Sim 5.1 + Isaac Lab 0.54 |
| Robot | Unitree G1 29-DoF (URDF) |
| RL Framework | RSL-RL 5.0 (PPO, on-policy) |
| Control | PD position control (implicit actuator model) |
| Python | 3.11 |

---

## Setup

### Prerequisites

- NVIDIA GPU (≥ 8 GB VRAM recommended)
- [Isaac Sim 5.1](https://developer.nvidia.com/isaac-sim) installed at `~/isaacsim`
- [Isaac Lab 0.54](https://github.com/isaac-sim/IsaacLab) installed at `~/IsaacLab`
- Conda environment `env_isaaclab` set up per Isaac Lab instructions

### Install

```bash
git clone <this-repo>
cd unitree_rl_lab
conda activate env_isaaclab
./unitree_rl_lab.sh --install
```

---

## Training

### Base locomotion (flat terrain)

```bash
conda activate env_isaaclab
cd unitree_rl_lab

# headless (fast, recommended for full training)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --num_envs 1024

# with GUI
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --num_envs 1024
```

Checkpoints saved every 100 iterations to `logs/rsl_rl/unitree_g1_29dof_velocity/<timestamp>/`.

### Sim-to-real fine-tuning (resume from base policy)

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity-Sim2Real \
  --num_envs 1024 \
  --resume \
  --checkpoint logs/rsl_rl/unitree_g1_29dof_velocity/<timestamp>/model_25600.pt
```

---

## Evaluation

### Standard play (no disturbance)

```bash
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity-Sim2Real \
  --load_run <timestamp>
```

### Push-robustness test (single push)

```bash
python scripts/rsl_rl/play_push.py \
  --task Unitree-G1-29dof-Velocity-Sim2Real \
  --load_run <timestamp> \
  --num_envs 16 --real_time \
  --push_force 300 \     # max force per axis (N), direction random horizontal
  --push_interval 2.0 \  # seconds between pushes
  --push_duration 0.2    # seconds each push is held
```

### Cyclic push schedule (progressive stress test)

```bash
python scripts/rsl_rl/play_cycle.py \
  --task Unitree-G1-29dof-Velocity-Sim2Real \
  --load_run <timestamp> \
  --num_envs 16 --real_time \
  --phase_vel_x 1.0 0.5 0.5 0.5   # forward velocity per phase (m/s)
  # Cycle: 10s no push → 10s @ 100 N → 10s @ 200 N → 10s @ 300 N → repeat
```

---

## Technical Summary

### 1. Simulation Setup

The environment uses Isaac Lab's `ManagerBasedRLEnv` with:

- **Physics timestep**: 5 ms (200 Hz), **policy frequency**: 50 Hz (decimation = 4)
- **Episode length**: 20 s
- **Terrain**: Procedural curriculum (flat → stairs → slopes → rough), 9 × 21 grid
- **Actuators**: Implicit PD model, per-joint stiffness/damping from URDF spec

### 2. Observation Space (actor, proprioceptive only)

| Term | Dim | Noise |
|---|---|---|
| Base angular velocity | 3 | Uniform ±0.3 |
| Projected gravity | 3 | Uniform ±0.1 |
| Velocity command (vx, vy, ωz) | 3 | — |
| Joint position (relative to default) | 29 | Uniform ±0.02 rad |
| Joint velocity | 29 | Uniform ±2.5 rad/s |
| Last action | 29 | — |
| **History** | × 5 frames | — |
| **Total** | **480** | — |

The critic receives an additional privileged observation (`base_lin_vel`, 3 dims) — asymmetric actor-critic design to keep the actor deployable without velocity estimation on real hardware.

### 3. Action Space

29-dimensional continuous joint position targets, clipped to joint limits and tracked by the PD controller.

### 4. Reward Shaping

19 reward terms are used, broadly split into three groups:

**Locomotion incentives** (positive):

| Term | Weight | Purpose |
|---|---|---|
| `track_lin_vel_xy` | +1.0 | Track commanded forward/lateral velocity |
| `track_ang_vel_z` | +0.5 | Track yaw rate command |
| `alive` | +0.15 | Staying upright penalty-free |
| `gait` | +0.5 | Encourage alternating foot contacts |
| `feet_clearance` | +1.0 | Reward foot swing height |

**Stability penalties** (negative):

| Term | Weight | Purpose |
|---|---|---|
| `flat_orientation_l2` | -5.0 | Penalise tilting torso |
| `base_height` | -10.0 | Maintain nominal pelvis height (0.78 m) |
| `base_linear_velocity` (z) | -2.0 | Suppress vertical bouncing |
| `base_angular_velocity` (xy) | -0.05 | Suppress roll/pitch rate |
| `undesired_contacts` | -1.0 | Penalise knee/shin contact |
| `feet_slide` | -0.2 | Penalise foot slip |

**Effort / smoothness penalties** (negative):

| Term | Weight | Purpose |
|---|---|---|
| `joint_vel` | -0.001 | Low joint speed preference |
| `joint_acc` (clipped) | -2.5×10⁻⁷ | Smooth acceleration; clipped to avoid physics blowup |
| `action_rate` | -0.05 | Smooth action changes between steps |
| `energy` (clipped) | -2×10⁻⁵ | Torque × velocity efficiency |
| `dof_pos_limits` | -5.0 | Stay within joint range |
| `joint_deviation_arms/waists/legs` | -0.1 to -1.0 | Arms/waist stay near neutral |

> **Design note — reward clipping**: under aggressive domain randomization, `joint_acc` and `energy` can grow unbounded in physically unstable configurations, poisoning the PPO replay buffer. Both terms are clamped per-step before weighting (max raw values: energy ≤ 50, joint_acc_l2 ≤ 2×10⁷). This prevents a single unstable environment instance from corrupting training.

### 5. Curriculum

Two concurrent curricula:

- **Terrain difficulty**: advances when the average robot displacement exceeds a threshold; retreats on failure. Levels progress from flat ground to rough/stepped terrain.
- **Velocity command**: begins at ±0.1 m/s and linearly advances to the limit range (vx: −0.5 → +1.0 m/s, vy: ±0.3 m/s, ωz: ±0.2 rad/s) as the robot demonstrates consistent tracking.

### 6. Domain Randomization (Sim-to-Real)

Applied at environment startup to each instance independently:

| Parameter | Range | Mode |
|---|---|---|
| Static / dynamic friction | [0.2, 1.2] | startup |
| Torso added mass | [−2, +5] kg | startup |
| All link mass scale | [0.9, 1.1] × | startup |
| Torso CoM offset | ±3 cm (xy), ±2 cm (z) | startup |
| PD stiffness / damping | [0.85, 1.15] × nominal | startup |
| Joint friction / armature | [0.8, 1.2] × / [0.9, 1.1] × | startup |
| External impulse (reset) | ±50 N force, ±5 Nm torque | each reset |
| Push velocity | ±1.0 m/s (x/y), ±0.5 rad/s (yaw) | every 3–6 s |

### 7. Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | PPO (on-policy) |
| Steps per env per update | 24 |
| Mini-batches | 4 |
| PPO epochs per update | 5 |
| Learning rate | 1×10⁻³ (adaptive KL, target 0.01) |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Entropy coefficient | 0.01 |
| Clip parameter ε | 0.2 |
| Max gradient norm | 1.0 |
| Actor / critic architecture | MLP [512, 256, 128], ELU |
| Output distribution | Gaussian (init std = 1.0) |
| Training environments | 1024 |
| Total training steps (base) | ~26M (25 600 iters × 1024 envs × 24 steps) |

### 8. Training Procedure

1. **Base training** (~8 h, 25 600 iterations on a single RTX 4060): policy learns stable walking on flat terrain; curriculum gradually advances terrain difficulty and command velocity.
2. **Sim-to-real fine-tuning** (resume from checkpoint): domain randomization is applied; policy adapts over ~3 500 iterations before re-stabilizing.

Key metrics at end of base training: mean reward ≈ 40, episode length ≈ 1 000 steps (full episodes, no falls), terrain level ≈ 4.8.

### 9. Push-Recovery Results

Under the cyclic push schedule (100 / 200 / 300 N random horizontal impulses every 2 s):
- The policy maintains balance against pushes up to **~300 N** (≈ 8.5 m/s² horizontal acceleration on a 35 kg robot).
- Recovery time from a 300 N impulse is typically **< 1 s**.

---

## Repository Structure

```
unitree_rl_lab/
├── scripts/rsl_rl/
│   ├── train.py           # Training entry point
│   ├── play.py            # Standard evaluation
│   ├── play_push.py       # Robustness test with random pushes
│   └── play_cycle.py      # Cyclic push schedule evaluation
├── source/unitree_rl_lab/unitree_rl_lab/
│   ├── assets/robots/unitree.py          # Robot asset config (G1, Go2, H1)
│   └── tasks/locomotion/
│       ├── agents/rsl_rl_ppo_cfg.py      # PPO hyperparameters
│       ├── mdp/
│       │   ├── rewards.py                # Custom reward functions + clipped wrappers
│       │   └── observations.py           # Custom observation terms
│       └── robots/g1/29dof/
│           ├── velocity_env_cfg.py           # Base locomotion env
│           └── velocity_sim2real_env_cfg.py  # Sim-to-real DR override
└── logs/rsl_rl/                          # Training checkpoints (gitignored)
```

---

## Notes

- Checkpoints and logs are gitignored. Download pretrained checkpoints from `deploy/robots/g1_29dof/config/policy/velocity/v0/exported/` (shipped with repo) or train from scratch.
- All modifications to the base `unitree_rl_lab` repo are backward-compatible; the original `Unitree-G1-29dof-Velocity` task is unchanged.
