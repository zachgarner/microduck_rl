"""Microduck headstand task (v1) — standing → fold forward → split-leg headstand, held.

Episodic trick. Triggered at deployment like standup/roulade (policy switch =
the entry starts). The policy folds forward, puts the flat top of its head on
the floor, extends one leg into a split, pushes off the last foot and swings
the trunk up with the neck, then balances on the head alone with the legs in
a V. Zach's spec (former acrobat, Sep 18 2026): "head to floor, legs in ...
a deep split, extend the leg as far as possible, but almost certainly
requiring a tinier calibrated hop with the leg touching the ground."

Physics that shaped this cfg (scripts/headstand/, docs/headstand/):
  • The hold pose HEADSTAND_OVERRIDES below balances statically on the
    allcollisions model: trunk_base z 0.117, ~20° off inverted-vertical, feet
    at 0.16 m. 15 of 30 noisy drops land, so active balance is required and
    the split legs are what provide it.
  • A foot cannot stay on the floor past ~60° toward inverted (leg 10 cm, hip
    15 cm up), so the last foot leaves at ~70° over and the swing is ~35–50°
    of trunk rotation about the head, driven by the neck (0.72 N·m gravity
    torque at horizontal vs 0.96 N·m servo cap) plus the foot push.

Design, following AGENTS.md and the standup/roulade lessons:
  • ONE dense swing signal: potential-based Δ(inverted_cos), support-gated,
    signed (rocking is a wash, ballistic flips pay nothing).
  • The hold is a product of Gaussians (height × inverted × pose) paid only
    with the head latched on the floor and both feet in the air. A flop
    scores ~0, a tripod is gated out. Broad stds + a sharp inverted layer.
  • Hard gates for what counts (state-based, not nudges): head-top latch,
    feet-off, airborne penalty (a jump is never the entry), other-body floor
    contact penalty (thigh/shin/trunk down = flop).
  • Reverse curriculum via the spawn mix: hold-heavy first (learn to balance),
    then partway (learn the swing), then standing-heavy (the whole entry).
  • Motion-blockers (body_ang_vel, angular_momentum) ≈ 0 during discovery;
    arrival damping / torque-rate polish introduced late by curriculum.
  • Symmetry OFF: the split is asymmetric (left leg forward). A right-leg
    version is a separate policy, like the two ball kicks.

Robot model: allcollisions — the head, thighs, shins and trunk all collide
with the floor, and the sensors below need to know which one touched.

Adversarial review before the first run (Sep 18 2026, 11 items) changed: the
contact solver budget (nconmax 200, iterations 30/50 like velstand/sitstand),
the partway spawn table (was putting 51% of spawns inside the floor), the
feet sensor (whole ankle bodies: the servo housing sits 2–8 mm below the sole
and made a free head+housing tripod), live-contact gating of every hold
reward, a flat-topped sharp term with the pose factor and a knee gate, a
head impact penalty with force sensing, a post-latch not-inverted tax, a
flatness gate on the swing, regulariser mass scaled to this task's ~6/step,
and curriculum stages stretched to roulade's proven pacing.
"""

import math
import os
from copy import deepcopy

# Symmetry — never for an asymmetric trick.
ENABLE_SYMMETRY = False

# ── Variant knobs, env-overridable for parallel runs (scripts/anyscale/variants.py)
PARK_TAX_WEIGHT      = float(os.environ.get("HEADSTAND_PARK_TAX", -0.5))     # per (1 - inverted_cos), always on
PROGRESS_WEIGHT      = float(os.environ.get("HEADSTAND_PROGRESS_W", 2.0))    # the swing potential
STANDING_PROB_STAGE0 = float(os.environ.get("HEADSTAND_STANDING_P0", 0.30))  # share of standing spawns at step 0
PARTWAY_PITCH_MIN_DEG = float(os.environ.get("HEADSTAND_PARTWAY_MIN_DEG", 95.0))  # partway spawns start here (the tripod)

# ── Domain randomisation (matched to standup/velocity for sim2real parity) ───
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # a shove mid-entry is incoherent (roulade)
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to the standup env) ───────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode: fold ~1.5 s + swing ~1 s + hold. 6 s leaves ~3 s of hold to pay.
EPISODE_LENGTH_S = 6.0

# Head impact: force on jaw_soft from the floor above this is billed. The
# static load in the hold is the whole 7.2 N body; the slam gate in mdp.py
# sits at 12 N, this fine starts at 10 N so a hard settle is priced too.
HEAD_IMPACT_THRESH_N = 10.0

# ── The hold pose (servo index → rad), measured Sep 18 2026 ───────────────────
# Hip yaw/roll, knees, ankles and head yaw/roll are 0 (NOT HOME: that is where
# the sweep found the balance). Left hip 1.2 rad is inside the 0.9 soft limit
# (1.41); the sweep's 1.6 rad pose sat on the hard limit and is not used. The
# neck at 1.0 rad is past its soft limit (0.94 of a 1.047 range): it costs
# dof_pos_limits ~0.06/step against a hold paying ~5/step, and every pose
# inside the soft limit landed half as often from noisy drops (9/30 vs 15/30).
# Sim2real note for later: 2.7° from the neck's hard stop.
HEADSTAND_OVERRIDES = {
    0:   0.0,   # left  hip_yaw
    1:   0.0,   # left  hip_roll
    2:   1.2,   # left  hip_pitch  (forward leg of the split)
    3:   0.0,   # left  knee
    4:   0.0,   # left  ankle
    5:   1.0,   # neck_pitch
    6:   1.25,  # head_pitch
    7:   0.0,   # head_yaw
    8:   0.0,   # head_roll
    9:   0.0,   # right hip_yaw
    10:  0.0,   # right hip_roll
    11:  0.8,   # right hip_pitch  (back leg of the split)
    12:  0.0,   # right knee
    13:  0.0,   # right ankle
}
# Two-leg variants (Zach, Sep 20 2026: "a double leg hop into headstand").
# tucked: the day-one sweep's knees-bent pose that balanced on its own
# (neck 1.0, head 1.5, hips 0.5, knees 1.5; trunk z 0.094 at rest).
# straight: legs together and straight up. No static balance exists for it
# (the day-one sweep found none), so the policy holds it actively.
HEADSTAND_TUCKED_OVERRIDES = {0: 0.0, 1: 0.0, 2: 0.5, 3: -1.5, 4: 0.0, 5: 1.0, 6: 1.5, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: -0.5, 12: 1.5, 13: 0.0}
HEADSTAND_STRAIGHT_OVERRIDES = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 1.25, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0}
HEADSTAND_TUCKED_Z = 0.094
# trunk_base z at rest in the hold pose (measured, allcollisions model).
HEADSTAND_Z = 0.117
# Hold spawns start here: the rest z plus 3 mm, so nothing falls in.
HEADSTAND_SPAWN_Z = HEADSTAND_Z + 0.003

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    HEAD_POSE_CMD_RESAMPLE_S,
    BODY_POSE_CMD_RESAMPLE_S,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_headstand_env_cfg(play: bool = False, kickup: bool = False, style: str = "split") -> ManagerBasedRlEnvCfg:
    """Create the Microduck headstand environment configuration.

    style: "split" (the default hold), "tucked" (two-leg hop, knees bent,
    legs together) or "straight" (two-leg hop, legs together straight up).

    kickup=True is the KICK-UP policy (Zach, Sep 20 2026: "The kick up policy
    is likely the one that matters"): episodes start head-down in the tripod
    or further along, never standing, and the policy learns the swing and the
    hold only. The fold from standing is another policy's job. Runs 1-5 with
    standing starts all parked in the tripod; this policy starts there.
    """

    # Whole ankle BODIES, not the sole geoms: each ankle carries three more
    # collision geoms (servo housing etc.) that sit 2–8 mm below the sole when
    # the ankle is pitched, and a sole-only sensor read "feet off" while the
    # housing carried the tripod (review item 3).
    if style == "fold":
        overrides, hold_z, knee_full, knee_zero = HEADSTAND_OVERRIDES, HEADSTAND_Z, 0.3, 0.6   # unused by the fold rewards
    elif style == "tucked":
        overrides, hold_z, knee_full, knee_zero = HEADSTAND_TUCKED_OVERRIDES, HEADSTAND_TUCKED_Z, 1.0, 1.3
    elif style == "straight":
        overrides, hold_z, knee_full, knee_zero = HEADSTAND_STRAIGHT_OVERRIDES, HEADSTAND_Z, 0.3, 0.6
    else:
        overrides, hold_z, knee_full, knee_zero = HEADSTAND_OVERRIDES, HEADSTAND_Z, 0.3, 0.6

    feet_ground_cfg = ContactSensorCfg(
        name=microduck_mdp._HEADSTAND_FEET_SENSOR,
        primary=ContactMatch(mode="body", pattern=r"^(ankle_left|ankle_right)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    # The head on the floor — the latch. jaw_soft carries the head collision
    # geoms; the latch additionally requires the flat top to be the part down.
    head_ground_cfg = ContactSensorCfg(
        name=microduck_mdp._HEADSTAND_HEAD_SENSOR,
        primary=ContactMatch(mode="body", pattern="^jaw_soft$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),   # force feeds the head impact penalty
        reduce="netforce",
        num_slots=1,
    )
    # Everything that is neither head nor foot: thighs, shins, hips, trunk,
    # neck. Any of these on the floor is a flop or a collapse.
    other_ground_cfg = ContactSensorCfg(
        name=microduck_mdp._HEADSTAND_OTHER_SENSOR,
        primary=ContactMatch(
            mode="body",
            pattern=r"^(?!jaw_soft$|yaw_roll_motion$|neck_pitch$|ankle_left$|ankle_right$).*",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    # Anything at all on the floor — the support gate (roulade lesson).
    robot_ground_cfg = ContactSensorCfg(
        name=microduck_mdp._HEADSTAND_SUPPORT_SENSOR,
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    # 70 collision geoms and a fold that piles head, thighs, shins and
    # self-contacts up at once: the base template's nconmax 35 / 10 solver
    # iterations overflow → NaN → nan_state resets that punish the fold
    # itself (sitstand found this; velstand carries the same budget).
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, head_ground_cfg, other_ground_cfg, robot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity", "track_angular_velocity", "air_time",
        "foot_clearance", "foot_swing_height", "foot_slip", "pose", "upright",
    ]:
        cfg.rewards.pop(name, None)

    # ── Rewards: the headstand set ────────────────────────────────────────────
    # Task mass at the hold ≈ 5.5/step (composite 4 + sharp 1.5); the swing
    # pays ~2/step × 2 for ~1 s. That is about half of standup's ~12, so the
    # shared regularisers below are scaled by ~0.46 to act at the relative
    # strength that transferred (AGENTS.md: compare reward mass, not weights).
    cfg.rewards["headstand_progress"] = RewardTermCfg(
        func=microduck_mdp.headstand_progress,
        weight=PROGRESS_WEIGHT,
    )
    cfg.rewards["headstand_composite"] = RewardTermCfg(
        func=microduck_mdp.headstand_composite,
        weight=4.0,
        params={
            "target_height":    hold_z,
            "height_std":       0.03,   # 3 cm: a flop at z≈0.06 scores ~0.03
            "inverted_std":     0.6,    # 1-cos(35°)=0.18 → e^-0.5; the balance basin scores visibly
            "pose_std":         0.45,   # joint-RMS, broad: partial pose still scores
            "target_overrides": overrides,
            "knee_full":        knee_full,
            "knee_zero":        knee_zero,
            "style":            style,
        },
    )
    # Flat-topped inside the measured 20° rest tilt, Gaussian beyond it:
    # 1-cos(30°)-0.06 = 0.074 → 0.44 at 30°, 0.06 at 40°. Carries the pose
    # factor and the knee gate, so it pays only the split.
    cfg.rewards["headstand_inverted_sharp"] = RewardTermCfg(
        func=microduck_mdp.headstand_inverted_sharp,
        weight=1.5,
        params={
            "inverted_std":     0.3,
            "pose_std":         0.45,
            "target_overrides": overrides,
            "knee_full":        knee_full,
            "knee_zero":        knee_zero,
            "style":            style,
        },
    )
    # Park tax, ALWAYS on (run 1 parked in a beak-down tripod the latched
    # version never saw): standing costs 1.0/step, a tripod 0.5/step, the
    # headstand 0. Same role as standup's height_stand_l1.
    cfg.rewards["headstand_not_inverted"] = RewardTermCfg(
        func=microduck_mdp.headstand_not_inverted_tax,
        weight=PARK_TAX_WEIGHT,
    )
    # Gates — hard, from step 0, cheap to compute, impossible to farm.
    cfg.rewards["headstand_feet_down"] = RewardTermCfg(
        func=microduck_mdp.headstand_feet_down_penalty,
        weight=-1.0,
        params={"inverted_lo": 0.5, "inverted_hi": 0.82},
    )
    # A flop is a TERMINATION (below), not a per-step bill: kick-up run 1 froze
    # in the tripod because a failed attempt that flopped cost -1/step for the
    # rest of the episode while freezing cost -0.5. A small one-off cost
    # remains so a flop is still worse than a clean miss.
    cfg.rewards["headstand_other_contact"] = RewardTermCfg(
        func=microduck_mdp.headstand_other_contact_penalty,
        weight=-0.2,
    )
    cfg.rewards["headstand_airborne"] = RewardTermCfg(
        func=microduck_mdp.headstand_airborne_penalty,
        weight=-2.0,
    )
    # |a_z| impact shaping from step 0 (roulade lesson: style is the scarce
    # resource here, discovery is helped by the spawn mix). SELF-NEGATING →
    # POSITIVE weight (returns -|a_z|; a negative weight would pay for shocks).
    cfg.rewards["gentle"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,   # ramped to 0.005 by curriculum, roulade's schedule
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    # Head impact: N above threshold on jaw_soft from the floor (velstand's
    # body_impact_cost). The arrival jackpot buys violence unless planting the
    # head hard costs something the trunk accelerometer alone cannot see.
    # Run 1 slammed at a median 21 N against a 15 N / 0.02 per N fine (0.12
    # once, vs a ~5.5/step hold). The structural fix is the slam gate in
    # _headstand_arrived_gently (a slam forfeits the hold for the episode);
    # this fine is raised so the steps it does apply to still hurt.
    cfg.rewards["head_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-1.0,
        params={"sensor_name": head_ground_cfg.name, "threshold": HEAD_IMPACT_THRESH_N},
    )
    # Whip tax above the measured natural tumble band (roulade run-4: p90
    # transit 5.5 rad/s; a swing of ~45° has no business above 6).
    cfg.rewards["headstand_overspeed"] = RewardTermCfg(
        func=microduck_mdp.roulade_overspeed_penalty,
        weight=-0.1,
        params={"omega_max": 6.0},
    )
    # Arrival damper — wobble around the balance point only. Starts at 0,
    # curriculum below (standup: timing, not magnitude, protects discovery).
    cfg.rewards["headstand_arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.headstand_arrival_damping,
        weight=0.0,
        params={"inverted_full": 0.94, "inverted_zero": 0.71},
    )

    if style == "fold":
        # The fold policy: standing → resting pike. Drop every headstand hold
        # term and pay the pike instead; keep the gates that price slams,
        # jumps and flops. Terminates on a flop like the headstand.
        pike_q = microduck_mdp._HEADSTAND_PIKE_QPOS
        pike_pitch = float(microduck_mdp._HEADSTAND_TRIPOD_PITCH)
        pike_overrides = {i: float(pike_q[7 + i]) for i in range(14)}
        for name in ("headstand_progress", "headstand_composite", "headstand_inverted_sharp",
                     "headstand_not_inverted", "headstand_feet_down", "headstand_arrival_damping"):
            cfg.rewards.pop(name, None)
        cfg.rewards["fold_progress"] = RewardTermCfg(
            func=microduck_mdp.fold_progress, weight=2.0, params={"target_pitch": pike_pitch},
        )
        cfg.rewards["fold_composite"] = RewardTermCfg(
            func=microduck_mdp.fold_composite, weight=4.0,
            params={
                "target_pitch": pike_pitch, "pitch_std": 0.25,            # ≈ 14°
                "target_height": float(pike_q[2]), "height_std": 0.03,
                "pose_std": 0.45, "target_overrides": pike_overrides,
            },
        )
        cfg.rewards["fold_overshoot"] = RewardTermCfg(
            func=microduck_mdp.fold_overshoot_penalty, weight=-1.0, params={"target_pitch": pike_pitch},
        )
        # Standing-still tax: every step short of the pike angle costs a little
        # (standup's height-L1 lesson), so standing there is not free.
        cfg.rewards["fold_not_folded"] = RewardTermCfg(
            func=microduck_mdp.headstand_not_inverted_tax, weight=-0.25,
        )

    # ── Sim2real regularisers (velocity's set; motion-blockers kept ≈ 0) ─────
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002   # the swing IS ω
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)
    # Light: the fold brings the head near the chest and the legs near the trunk.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (identical layout to walking / standup policies) ─────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── Commands: every slot alive at a tiny range, none tracked ─────────────
    # The headstand takes no command; the 13D command block stays in the obs
    # (61D contract) with small non-zero sampling so no input neuron is dead.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)),
    )
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        ranges=((-0.005, 0.005),) * 3 + ((-0.05, 0.05),) * 3,
    )
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "body_pose"},
        )
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ──────────────────────────────────────────────────────────
    # Inverted is the goal, so the tilt-based fall termination does not apply.
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )
    cfg.terminations["flopped"] = TerminationTermCfg(
        func=microduck_mdp.headstand_flopped,
        time_out=False,
        params={"grace_steps": 25},
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields, mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history, mode="reset",
    )
    if style == "fold":
        cfg.events["reset_fold_progress"] = EventTermCfg(
            func=microduck_mdp.reset_fold_progress, mode="reset",
        )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Spawn mix — stage 0 of the reverse curriculum below: mostly the hold.
    standing_p0 = 0.0 if kickup else (0.7 if style == "fold" else STANDING_PROB_STAGE0)
    # Kick-up: 40% of episodes start in the RESTING tripod (measured settled
    # state, head and feet on the floor from step 0), 30% partway through the
    # swing, 30% in the hold.
    partway_lerp = (0.0, 1.0) if kickup else (0.4, 1.0)
    partway_min_deg = 100.0 if kickup else PARTWAY_PITCH_MIN_DEG
    cfg.events["set_headstand_spawn"] = EventTermCfg(
        func=microduck_mdp.reset_headstand_spawn,
        mode="reset",
        params={
            # Full task (kickup=False), after kick-up run 3 landed the hop from
            # the pike (31/32, Sep 20 2026): standing starts learn the bow into
            # the pike, pike starts keep the hop warm, the rest polish.
            "standing_prob":       standing_p0,
            "partway_prob":        0.30 if kickup else (0.0 if style == "fold" else 0.20),
            "hold_prob":           0.30 if kickup else (0.0 if style == "fold" else 0.15),
            "tripod_prob":         0.40 if kickup else (0.30 if style == "fold" else 0.35),
            "standing_z_min":      0.11,
            "standing_z_max":      0.12,
            "standing_tilt_max":   math.radians(3.0),
            # Run 4 (Sep 20 2026): every variant parked in the ~95° tripod and
            # no episode had ever started between 95° and 125°, so the first
            # half of the swing had no on-policy data (AGENTS.md reverse-
            # curriculum rule). Partway now starts AT the tripod. Spawns
            # below ~122° are born un-latched (the head top is not down yet)
            # and latch as they rotate.
            "partway_pitch_min":   math.radians(partway_min_deg),
            "partway_pitch_max":   math.radians(165.0),
            "partway_lerp_range":  partway_lerp,
            "hold_pitch_noise":    math.radians(8.0),
            "hold_z":              hold_z + 0.003,
            "hold_overrides":      overrides,
            "joint_noise_std":     0.05,
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )
    cfg.events.pop("push_robot", None)

    # ── Terrain: flat only ────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curricula ─────────────────────────────────────────────────────────────
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)

    # Reverse curriculum on the spawn mix: balance first, then the swing, then
    # the whole entry from standing. Pacing copied from the roulade's proven
    # stages (its run 2 shifted away from mid-roll before standing-spawn rolls
    # were mastered and lost the skill): 1500 / 3000 / 5000, not 300 / 600 /
    # 1000. If a metric steps DOWN at a boundary, stretch, never move earlier.
    cfg.curriculum["headstand_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_headstand_spawn",
            "param_stages": [
                {"step": 0,         "params": {"standing_prob": 0.0, "partway_prob": 0.30, "hold_prob": 0.30, "tripod_prob": 0.40}},
                {"step": 1500 * 24, "params": {"standing_prob": 0.0, "partway_prob": 0.25, "hold_prob": 0.20, "tripod_prob": 0.55}},
                {"step": 3000 * 24, "params": {"standing_prob": 0.0, "partway_prob": 0.20, "hold_prob": 0.15, "tripod_prob": 0.65}},
            ] if kickup else [
                {"step": 0,         "params": {"standing_prob": 0.7, "partway_prob": 0.0, "hold_prob": 0.0, "tripod_prob": 0.3}},
                {"step": 1000 * 24, "params": {"standing_prob": 0.8, "partway_prob": 0.0, "hold_prob": 0.0, "tripod_prob": 0.2}},
            ] if style == "fold" else [
                {"step": 0,         "params": {"standing_prob": STANDING_PROB_STAGE0, "partway_prob": 0.20, "hold_prob": 0.15, "tripod_prob": 0.35}},
                {"step": 1000 * 24, "params": {"standing_prob": 0.45, "partway_prob": 0.15, "hold_prob": 0.10, "tripod_prob": 0.30}},
                {"step": 2000 * 24, "params": {"standing_prob": 0.60, "partway_prob": 0.10, "hold_prob": 0.05, "tripod_prob": 0.25}},
            ],
        },
    )
    # The park tax is on from step 0 and tightens once hold spawns balance.
    if style != "fold":
        cfg.curriculum["not_inverted_weight"] = CurriculumTermCfg(
                func=microduck_mdp.reward_weight,
                params={
                    "reward_name": "headstand_not_inverted",
                    "weight_stages": [
                        {"step": 0,          "weight": PARK_TAX_WEIGHT},
                        {"step": 1000 * 24,  "weight": PARK_TAX_WEIGHT * 1.5},
                        {"step": 2000 * 24,  "weight": PARK_TAX_WEIGHT * 2.0},
                    ],
                },
            )
        cfg.curriculum["gentle_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "gentle",
                "weight_stages": [
                    {"step": 0,          "weight": 0.002},
                    {"step": 1500 * 24,  "weight": 0.0035},
                    {"step": 3000 * 24,  "weight": 0.005},
                ],
            },
        )
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )
    # Smoothness: velocity's action_rate ramp scaled by this task's reward
    # mass (5.5/12 ≈ 0.46), stretched to the spawn-mix pacing; the polish
    # terms come in at 2500 like the roulade, after the skill exists.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.05},
                {"step": 1000 * 24,  "weight": -0.1},
                {"step": 1500 * 24,  "weight": -0.2},
                {"step": 2000 * 24,  "weight": -0.3},
                {"step": 2500 * 24,  "weight": -0.4},
                {"step": 3000 * 24,  "weight": -0.46},
            ],
        },
    )
    if style != "fold":
        cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
                func=microduck_mdp.reward_weight,
                params={
                    "reward_name": "headstand_arrival_damping",
                    "weight_stages": [
                        {"step": 0,          "weight": 0.0},
                        {"step": 2500 * 24,  "weight": -0.025},
                        {"step": 3500 * 24,  "weight": -0.05},
                    ],
                },
            )
        cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "joint_torque_rate_l2",
                "weight_stages": [
                    {"step": 0,          "weight": 0.0},
                    {"step": 2500 * 24,  "weight": -1e-3},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckHeadstandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_headstand",
    run_name="microduck_headstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6_000,
)

def _two_leg_runner_cfg(name: str) -> RslRlOnPolicyRunnerCfg:
    """Two-leg kick-up variants: symmetric trick, so Pollen's mirror loss is ON."""
    from dataclasses import replace
    algo = replace(MicroduckHeadstandRlCfg.algorithm, symmetry_cfg=SYMMETRY_CFG)
    return RslRlOnPolicyRunnerCfg(
        actor=MicroduckHeadstandRlCfg.actor, critic=MicroduckHeadstandRlCfg.critic, algorithm=algo,
        wandb_project="mjlab_microduck", experiment_name=name, run_name=name,
        save_interval=250, num_steps_per_env=24, max_iterations=6_000,
    )


MicroduckHeadstandKickupTuckedRlCfg = _two_leg_runner_cfg("microduck_headstand_kickup_tucked")
MicroduckHeadstandFoldRlCfg = _two_leg_runner_cfg("microduck_headstand_fold")   # the bow is symmetric too
MicroduckHeadstandKickupStraightRlCfg = _two_leg_runner_cfg("microduck_headstand_kickup_straight")

# The kick-up policy: same network, its own experiment so checkpoints and
# wandb runs never mix with the full-entry task.
MicroduckHeadstandKickupRlCfg = RslRlOnPolicyRunnerCfg(
    actor=MicroduckHeadstandRlCfg.actor,
    critic=MicroduckHeadstandRlCfg.critic,
    algorithm=MicroduckHeadstandRlCfg.algorithm,
    wandb_project="mjlab_microduck",
    experiment_name="microduck_headstand_kickup",
    run_name="microduck_headstand_kickup",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6_000,
)
