"""Microduck back-roll exit: from the headstand hold, roll down the back and stand.

Zach, Sep 20 2026 night: "fold -> kick up to straight -> hold -> roll -> ...".
The exit from a headstand is the second half of Pollen's forward roll: the
roll passes through "inverted, 180°" on its way from standing to standing,
and the roulade env already trains from mid-roll spawns (its reverse
curriculum). This task IS the roulade env with every episode spawned at
180° in our hold pose (straight or split) and the rotation accumulator
pre-set to 180°, so the roll's own progress, landing and stand-tax rewards
carry it to standing. No new reward terms.

Runtime chaining: the runtime hot-swaps ONNX policies with a shared obs
contract (AGENTS.md), so hold → backroll is a policy switch, like the roll
handing to the standing policy after landing.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from mjlab_microduck.tasks.microduck_headstand_env_cfg import (
    HEADSTAND_OVERRIDES,
    HEADSTAND_STRAIGHT_OVERRIDES,
    HEADSTAND_Z,
)
from mjlab.rl import RslRlOnPolicyRunnerCfg


def make_microduck_backroll_env_cfg(play: bool = False, style: str = "straight") -> ManagerBasedRlEnvCfg:
    """style: straight (legs together), split (the split hold, legs free during
    the roll: it learned a tuck roll), splitover (the split hold, legs kept
    split and straight while the trunk goes over; Zach, Sep 21 2026: "The
    point is splitting over")."""
    cfg = make_microduck_roulade_env_cfg(play=play)
    overrides = HEADSTAND_STRAIGHT_OVERRIDES if style == "straight" else HEADSTAND_OVERRIDES
    # Same collision model as every other headstand policy (run 1 trained on the
    # roll's feet-and-shell model and lost 6 of 15 in the chained routine that
    # runs on allcollisions); same solver budget as the headstand tasks.
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    spawn = cfg.events["set_roulade_state"].params
    spawn.update(
        standing_prob=0.0,
        midroll_prob=1.0,
        # Wide enough to cover what the hold policy actually hands over.
        midroll_pitch_min=math.radians(165.0),
        midroll_pitch_max=math.radians(190.0),
        # trunk_base rests at HEADSTAND_Z in the hold; spawn a hair above.
        midroll_z_min=HEADSTAND_Z + 0.002,
        midroll_z_max=HEADSTAND_Z + 0.006,
        midroll_omega_range=(0.0, 1.0),        # a held headstand wobbles a little
        tuck_overrides=overrides,
        tuck_factor_range=(0.9, 1.0),          # the hold pose, nearly (+ noise)
        joint_noise_std=0.08,
    )
    # The roulade's spawn-mix curriculum would reintroduce standing spawns; pin it.
    cfg.curriculum["roulade_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_roulade_state",
            "param_stages": [{"step": 0, "params": {"standing_prob": 0.0, "midroll_prob": 1.0}}],
        },
    )
    # The exit is a controlled fall onto the back plus the rise: a 6 s episode
    # like the headstand's (the roulade's 5 s starts standing).
    cfg.episode_length_s = 6.0
    if style == "splitover":
        from mjlab.managers import RewardTermCfg
        # Progress pays only while the legs are split and straight during the
        # over-the-top window; the tuck costs per step in the same window
        # (~2/step against the progress term's ~5.6/step, so a tuck roll
        # nets less than a slow split-over). Same rate cap as the roll.
        cfg.rewards["roulade_progress"] = RewardTermCfg(
            func=microduck_mdp.roulade_progress_split,
            weight=8.0,
            params={"target_angle": 2 * math.pi, "max_paid_rate": 5.0,
                    "angle_lo": math.radians(170.0), "angle_hi": math.radians(330.0)},
        )
        cfg.rewards["roulade_tuck"] = RewardTermCfg(
            func=microduck_mdp.roulade_tuck_penalty,
            weight=-2.0,
            params={"angle_lo": math.radians(170.0), "angle_hi": math.radians(330.0)},
        )
    return cfg


def _runner(name: str) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=MicroduckRouladeRlCfg.actor, critic=MicroduckRouladeRlCfg.critic,
        algorithm=MicroduckRouladeRlCfg.algorithm,
        wandb_project="mjlab_microduck", experiment_name=name, run_name=name,
        save_interval=250, num_steps_per_env=24, max_iterations=6_000,
    )


MicroduckBackrollStraightRlCfg = _runner("microduck_headstand_backroll_straight")
MicroduckBackrollSplitRlCfg = _runner("microduck_headstand_backroll_split")
MicroduckSplitOverRlCfg = _runner("microduck_headstand_splitover")
