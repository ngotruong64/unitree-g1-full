"""G1-29dof velocity tracking with stronger domain randomization for sim2real."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import (
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
    RobotPlayEnvCfg,
)


@configclass
class Sim2RealEventCfg(EventCfg):
    """Domain-randomization config tuned for sim2real transfer.

    Resume-compatible: does not change observation or action dimensions.
    """

    # --- physics material: widen the friction range slightly (real floors vary a lot) ---
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.2),
            "dynamic_friction_range": (0.2, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # --- mass: widen torso range AND randomize every link (carrying payload, accessory mass) ---
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-2.0, 5.0),
            "operation": "add",
        },
    )

    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # --- center-of-mass offset on the torso (assembly tolerance / sensors / battery position) ---
    randomize_torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.02, 0.02)},
        },
    )

    # --- PD gains: real motors have different stiffness/damping vs the nominal config ---
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # --- joint friction / armature / damping (drivetrain stiction, gear inertia) ---
    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # --- random external impulse on the base at every reset (simulates startup wobble / collisions) ---
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (-50.0, 50.0),
            "torque_range": (-5.0, 5.0),
        },
    )

    # --- broader joint reset to force the policy to recover from unusual poses ---
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (-1.5, 1.5),
        },
    )

    # --- stronger and faster pushes (real environments are not gentle) ---
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-0.5, 0.5)}},
    )


@configclass
class Sim2RealObservationsCfg(ObservationsCfg):
    """Higher observation noise to model real IMU/encoder noise."""

    def __post_init__(self):  # type: ignore[override]
        # Boost noise on the actor's proprioceptive obs. Critic (privileged) stays clean.
        self.policy.base_ang_vel.noise = Unoise(n_min=-0.3, n_max=0.3)
        self.policy.projected_gravity.noise = Unoise(n_min=-0.1, n_max=0.1)
        self.policy.joint_pos_rel.noise = Unoise(n_min=-0.02, n_max=0.02)
        self.policy.joint_vel_rel.noise = Unoise(n_min=-2.5, n_max=2.5)


@configclass
class Sim2RealRewardsCfg(RewardsCfg):
    """Override unbounded reward terms with clipped versions to prevent physics blowup."""

    energy = RewTerm(func=mdp.energy_clipped, weight=-2e-5)
    joint_acc = RewTerm(func=mdp.joint_acc_l2_clipped, weight=-2.5e-7)


@configclass
class RobotSim2RealEnvCfg(RobotEnvCfg):
    """G1-29dof velocity env with sim2real-grade domain randomization."""

    events: Sim2RealEventCfg = Sim2RealEventCfg()
    observations: Sim2RealObservationsCfg = Sim2RealObservationsCfg()
    rewards: Sim2RealRewardsCfg = Sim2RealRewardsCfg()


@configclass
class RobotSim2RealPlayEnvCfg(RobotPlayEnvCfg):
    """Play config mirroring the sim2real env."""

    events: Sim2RealEventCfg = Sim2RealEventCfg()
    observations: Sim2RealObservationsCfg = Sim2RealObservationsCfg()
    rewards: Sim2RealRewardsCfg = Sim2RealRewardsCfg()
