"""Headstand cfg invariants (CPU, no GPU). Locks the physics-derived constants,
the reward sign convention, the gate wiring and the 61D obs contract."""

import math

import mujoco
import numpy as np
import pytest

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_headstand_env_cfg import (
    HEADSTAND_OVERRIDES,
    HEADSTAND_Z,
    MicroduckHeadstandRlCfg,
    make_microduck_headstand_env_cfg,
)
from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_XML


def test_uses_the_allcollisions_model():
    # The gates need to know WHICH body touched the floor (thigh, shin, trunk),
    # and those bodies only collide on the allcollisions model.
    cfg = make_microduck_headstand_env_cfg()
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
    assert cfg.scene.entities["robot"] is MICRODUCK_ALLCOLLISIONS_ROBOT_CFG


def test_hold_pose_is_inside_the_hard_joint_limits():
    # The sweep's 1.6 rad left hip sat ON the hard limit and is not used. The
    # neck (1.0 of 1.047) is the one joint past its 0.9 soft limit: that only
    # costs dof_pos_limits 0.06/step against a hold paying ~5/step, and the
    # in-soft-limit alternatives land half as often (Sep 18 2026 sweep).
    m = mujoco.MjModel.from_xml_path(str(MICRODUCK_ALLCOLLISIONS_XML))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    servo = [n for n in names if n != "trunk_base_freejoint"]
    assert len(servo) == 14 and len(HEADSTAND_OVERRIDES) == 14
    for idx, angle in HEADSTAND_OVERRIDES.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, servo[idx])
        lo, hi = m.jnt_range[jid]
        assert lo + 0.04 <= angle <= hi - 0.04, (servo[idx], angle, lo, hi)


def test_hold_pose_balances_on_the_head_alone():
    # AGENTS.md step 2, locked: drop the hold pose inverted, hold 3 s, and it
    # rests on jaw_soft alone with the trunk near inverted at HEADSTAND_Z.
    scene = MICRODUCK_ALLCOLLISIONS_XML.parent / "scene_allcollisions.xml"
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    joints = np.zeros(14)
    for idx, angle in HEADSTAND_OVERRIDES.items():
        joints[idx] = angle
    d.qpos[7:] = joints
    d.qpos[3:7] = [math.cos(math.pi / 2), 0.0, math.sin(math.pi / 2), 0.0]
    d.qpos[0:3] = [0.0, 0.0, 0.135]
    d.ctrl[:] = joints
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    up_z = d.xmat[1].reshape(3, 3)[2, 2]
    assert up_z < -0.85, up_z
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    touching = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if floor in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == floor else c.geom1
            touching.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[other]))
    assert touching == {"jaw_soft"}, touching
    trunk_z = d.xpos[1][2]
    assert abs(trunk_z - HEADSTAND_Z) < 0.01, trunk_z


def test_reward_signs_follow_the_convention():
    # mjlab-style costs (≥ 0) take a negative weight; the self-negating
    # trunk_vertical_accel_penalty takes a POSITIVE weight. A wrong sign here
    # pays for the violation and the policy farms it.
    cfg = make_microduck_headstand_env_cfg()
    for name in ("headstand_feet_down", "headstand_other_contact", "headstand_airborne",
                 "headstand_overspeed", "head_impact", "action_rate_l2", "body_ang_vel",
                 "angular_momentum", "self_collisions"):
        assert cfg.rewards[name].weight < 0.0, name
    assert cfg.rewards["headstand_not_inverted"].weight < 0.0  # always on since run 1's tripod park
    assert cfg.rewards["gentle"].weight > 0.0
    assert cfg.rewards["gentle"].func is microduck_mdp.trunk_vertical_accel_penalty
    for name in ("headstand_progress", "headstand_composite", "headstand_inverted_sharp"):
        assert cfg.rewards[name].weight > 0.0, name
    assert cfg.rewards["headstand_arrival_damping"].weight == 0.0  # curriculum introduces it
    assert "upright" not in cfg.rewards  # always-on upright would oppose the trick


def test_sensors_are_registered_under_the_names_mdp_reads():
    cfg = make_microduck_headstand_env_cfg()
    names = {s.name for s in cfg.scene.sensors}
    for expected in (microduck_mdp._HEADSTAND_HEAD_SENSOR, microduck_mdp._HEADSTAND_FEET_SENSOR,
                     microduck_mdp._HEADSTAND_OTHER_SENSOR, microduck_mdp._HEADSTAND_SUPPORT_SENSOR):
        assert expected in names, expected


def test_other_contact_pattern_excludes_head_and_feet_only():
    import re
    cfg = make_microduck_headstand_env_cfg()
    other = next(s for s in cfg.scene.sensors if s.name == microduck_mdp._HEADSTAND_OTHER_SENSOR)
    pat = re.compile(other.primary.pattern)
    for allowed in ("jaw_soft", "yaw_roll_motion", "neck_pitch", "ankle_left", "ankle_right"):
        assert not pat.match(allowed), allowed
    for flop in ("trunk_base", "upper_leg_left", "upper_leg_right", "leg", "leg_2", "neck", "hip_l"):
        assert pat.match(flop), flop


def test_spawn_mix_moves_from_hold_to_standing():
    cfg = make_microduck_headstand_env_cfg()
    stages = cfg.curriculum["headstand_spawn_mix"].params["param_stages"]
    hold = [s["params"]["hold_prob"] for s in stages]
    standing = [s["params"]["standing_prob"] for s in stages]
    assert hold == sorted(hold, reverse=True) and standing == sorted(standing)
    assert stages[0]["params"] == {k: cfg.events["set_headstand_spawn"].params[k] for k in stages[0]["params"]}
    assert cfg.events["set_headstand_spawn"].params["tripod_prob"] > 0.0   # the pike keeps the hop warm


def test_spawn_height_table_is_monotone_in_pitch_index():
    assert len(microduck_mdp._HEADSTAND_SPAWN_PITCH_DEG) == len(microduck_mdp._HEADSTAND_SPAWN_Z)
    assert float(microduck_mdp._HEADSTAND_SPAWN_PITCH_DEG[0]) == 90.0
    assert float(microduck_mdp._HEADSTAND_SPAWN_PITCH_DEG[-1]) == 180.0


def test_symmetry_is_off_for_the_asymmetric_split():
    assert MicroduckHeadstandRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckHeadstandRlCfg.experiment_name == "microduck_headstand"


def test_pushes_and_fall_termination_are_off():
    cfg = make_microduck_headstand_env_cfg()
    assert "push_robot" not in cfg.events
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_obs_parity_with_standup():
    # The ONNX must load in the runtime's policy slot: same term order as the
    # standup policy, group by group.
    from mjlab_microduck.tasks.microduck_standup_env_cfg import make_microduck_standup_env_cfg
    head = make_microduck_headstand_env_cfg()
    stand = make_microduck_standup_env_cfg()
    for grp in ("actor", "critic"):
        assert list(head.observations[grp].terms.keys()) == list(stand.observations[grp].terms.keys()), grp


# ── Review fixes, Sep 18 2026 ──────────────────────────────────────────────────

def test_contact_solver_budget_matches_the_full_collision_tasks():
    cfg = make_microduck_headstand_env_cfg()
    assert cfg.sim.nconmax >= 200
    assert cfg.sim.mujoco.iterations >= 30 and cfg.sim.mujoco.ls_iterations >= 50


def test_feet_sensor_covers_the_whole_ankle_bodies():
    # The servo housing on the ankle sits below the sole when pitched; a
    # sole-only sensor let a head + housing tripod pass every gate.
    cfg = make_microduck_headstand_env_cfg()
    feet = next(s for s in cfg.scene.sensors if s.name == microduck_mdp._HEADSTAND_FEET_SENSOR)
    assert feet.primary.mode == "body"
    import re
    assert re.fullmatch(feet.primary.pattern, "ankle_left") and re.fullmatch(feet.primary.pattern, "ankle_right")


def test_head_sensor_carries_force_for_the_impact_penalty():
    cfg = make_microduck_headstand_env_cfg()
    head = next(s for s in cfg.scene.sensors if s.name == microduck_mdp._HEADSTAND_HEAD_SENSOR)
    assert "force" in head.fields and "found" in head.fields
    assert cfg.rewards["head_impact"].params["sensor_name"] == head.name


def test_partway_spawns_start_above_the_floor():
    # Every (pitch, lerp, roll, noise) the spawn can draw must leave the lowest
    # collision corner above the floor. The first table put 51% inside it.
    import sys
    sys.path.insert(0, "scripts/headstand")
    from settle_sweep import lowest_point_z
    scene = MICRODUCK_ALLCOLLISIONS_XML.parent / "scene_allcollisions.xml"
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    cfg = make_microduck_headstand_env_cfg()
    p = cfg.events["set_headstand_spawn"].params
    target = np.zeros(14)
    for idx, angle in HEADSTAND_OVERRIDES.items():
        target[idx] = angle
    home = np.array([0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579, 0.0049, -0.4530])
    rng = np.random.default_rng(3)
    table_deg = microduck_mdp._HEADSTAND_SPAWN_PITCH_DEG.numpy()
    table_z = microduck_mdp._HEADSTAND_SPAWN_Z.numpy()
    below = 0
    for _ in range(300):
        pitch = rng.uniform(p["partway_pitch_min"], p["partway_pitch_max"])
        u = rng.uniform(*p["partway_lerp_range"])
        roll = rng.uniform(-math.radians(5), math.radians(5))
        q = home + u * (target - home) + rng.normal(0, p["joint_noise_std"], 14)
        z = np.interp(math.degrees(pitch), table_deg, table_z)
        cp, sp, cr, sr = math.cos(pitch / 2), math.sin(pitch / 2), math.cos(roll / 2), math.sin(roll / 2)
        d.qpos[7:] = q
        d.qpos[3:7] = [cr * cp, sr * cp, cr * sp, -sr * sp]
        d.qpos[0:3] = [0.0, 0.0, z]
        mujoco.mj_forward(m, d)
        below += lowest_point_z(m, d) < 0.0
    assert below == 0, f"{below} of 300 partway spawns start inside the floor"


def test_partway_spawns_begin_at_the_tripod():
    cfg = make_microduck_headstand_env_cfg()
    assert math.radians(90.0) <= cfg.events["set_headstand_spawn"].params["partway_pitch_min"] <= math.radians(100.0)
    assert abs(cfg.events["set_headstand_spawn"].params["hold_z"] - HEADSTAND_Z) < 0.005


def test_curricula_are_paced_like_the_roulade():
    cfg = make_microduck_headstand_env_cfg()
    mix = [s["step"] for s in cfg.curriculum["headstand_spawn_mix"].params["param_stages"]]
    # Warm-started from the kick-up policy (Sep 20 2026): the hop is known, so
    # standing starts ramp over 2000 iterations instead of 5000.
    assert mix[1] >= 1000 * 24 and mix[-1] >= 2000 * 24
    for name in ("arrival_damping_weight", "torque_rate_weight"):
        first_nonzero = next(s["step"] for s in cfg.curriculum[name].params["weight_stages"] if s["weight"] != 0.0)
        assert first_nonzero >= 2500 * 24, name
    ladder = [s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]]
    assert ladder[0] == cfg.rewards["action_rate_l2"].weight
    assert abs(ladder[-1]) <= 0.5  # standup's -1.0 scaled by this task's reward mass
    assert MicroduckHeadstandRlCfg.max_iterations >= 6000


def test_sharp_term_is_flat_inside_the_rest_tilt():
    # 1-cos(20°) is the plateau edge; a perfect vertical must not pay more
    # than the measured rest tilt does.
    assert abs(microduck_mdp._HEADSTAND_REST_TILT_COS - math.cos(math.radians(20.0))) < 1e-9
    cfg = make_microduck_headstand_env_cfg()
    params = cfg.rewards["headstand_inverted_sharp"].params
    assert params["target_overrides"] is HEADSTAND_OVERRIDES and params["knee_zero"] > params["knee_full"] > 0.0


def test_run1_lessons_park_tax_always_on_and_slam_gate():
    cfg = make_microduck_headstand_env_cfg()
    stages = cfg.curriculum["not_inverted_weight"].params["weight_stages"]
    assert stages[0]["weight"] == cfg.rewards["headstand_not_inverted"].weight < 0.0
    assert microduck_mdp._HEADSTAND_SLAM_N < 15.0 and microduck_mdp._HEADSTAND_SLAM_N > 7.2
    # a 21 N landing (run 1's median) must cost more than one step of the hold
    assert -cfg.rewards["head_impact"].weight * (21.0 - cfg.rewards["head_impact"].params["threshold"]) > 5.5


def test_forward_fold_gate_is_zero_for_a_backward_drop():
    # Variant A fell backwards onto its head top; the latch/swing/hold gates
    # must read 0 nose-up and 1 nose-down or vertical.
    import torch
    from types import SimpleNamespace
    def asset_with_pitch(deg):
        p = math.radians(deg)
        q = torch.tensor([[math.cos(p / 2), 0.0, math.sin(p / 2), 0.0]])
        return SimpleNamespace(data=SimpleNamespace(root_link_quat_w=q))
    g = microduck_mdp._forward_fold_gate
    assert float(g(asset_with_pitch(0))) == 1.0        # standing
    assert float(g(asset_with_pitch(100))) == 1.0      # forward fold, nose down
    assert float(g(asset_with_pitch(180))) == 1.0      # inverted
    assert float(g(asset_with_pitch(-100))) == 0.0     # fallen on the back
    assert float(g(asset_with_pitch(-135))) == 0.0


def test_kickup_never_spawns_standing_and_starts_in_the_tripod():
    cfg = make_microduck_headstand_env_cfg(kickup=True)
    p = cfg.events["set_headstand_spawn"].params
    assert p["standing_prob"] == 0.0
    assert math.radians(95.0) <= p["partway_pitch_min"] <= math.radians(105.0)   # partway starts just past the pike
    assert p["partway_lerp_range"][0] == 0.0
    for stage in cfg.curriculum["headstand_spawn_mix"].params["param_stages"]:
        assert stage["params"]["standing_prob"] == 0.0
    from mjlab_microduck.tasks.microduck_headstand_env_cfg import MicroduckHeadstandKickupRlCfg
    assert MicroduckHeadstandKickupRlCfg.experiment_name == "microduck_headstand_kickup"


def test_flop_terminates_and_swing_pays_the_frontier():
    cfg = make_microduck_headstand_env_cfg(kickup=True)
    assert "flopped" in cfg.terminations and cfg.terminations["flopped"].time_out is False
    assert cfg.rewards["headstand_other_contact"].weight > -0.5  # a one-off, not a per-step bill
    import inspect
    src = inspect.getsource(microduck_mdp.headstand_progress)
    assert "_headstand_max_inverted" in src and "clamp(capped - env._headstand_max_inverted, min=0.0)" in src


def test_kickup_spawns_in_the_pike():
    cfg = make_microduck_headstand_env_cfg(kickup=True)
    p = cfg.events["set_headstand_spawn"].params
    assert p["tripod_prob"] > 0.0 and p["standing_prob"] == 0.0
    assert math.radians(70) < microduck_mdp._HEADSTAND_TRIPOD_PITCH < math.radians(80)
    assert microduck_mdp._HEADSTAND_PIKE_QPOS.shape == (21,)
    assert 0.09 < float(microduck_mdp._HEADSTAND_PIKE_QPOS[2]) < 0.12


def test_two_leg_styles_are_symmetric_and_target_legs_together():
    from mjlab_microduck.tasks.microduck_headstand_env_cfg import (
        MicroduckHeadstandKickupTuckedRlCfg, MicroduckHeadstandKickupStraightRlCfg,
        HEADSTAND_TUCKED_OVERRIDES, HEADSTAND_STRAIGHT_OVERRIDES,
    )
    for style, rl in (("tucked", MicroduckHeadstandKickupTuckedRlCfg), ("straight", MicroduckHeadstandKickupStraightRlCfg)):
        cfg = make_microduck_headstand_env_cfg(kickup=True, style=style)
        assert cfg.rewards["headstand_composite"].params["style"] == style
        assert rl.algorithm.symmetry_cfg is not None
        ov = cfg.rewards["headstand_composite"].params["target_overrides"]
        assert abs(ov[2] + ov[11]) < 1e-9   # hips symmetric: no split
    assert abs(HEADSTAND_TUCKED_OVERRIDES[3]) >= 1.0 and HEADSTAND_STRAIGHT_OVERRIDES[3] == 0.0
    from mjlab_microduck.tasks.microduck_headstand_env_cfg import MicroduckHeadstandKickupRlCfg
    assert MicroduckHeadstandKickupRlCfg.algorithm.symmetry_cfg is None   # the split stays asymmetric


def test_fold_has_no_standing_tax_and_no_flop_termination():
    cfg = make_microduck_headstand_env_cfg(style="fold")
    assert "fold_not_folded" not in cfg.rewards and "flopped" not in cfg.terminations
    assert cfg.rewards["fold_progress"].weight >= 5.0 and cfg.rewards["fold_composite"].weight > 0
    assert cfg.rewards["headstand_other_contact"].weight < 0


def test_ramp_setpoint_starts_at_the_spawn_and_arrives_at_one():
    import torch
    from types import SimpleNamespace
    import importlib
    microduck_mdp._HEADSTAND_RAMP_S = 2.0
    env = SimpleNamespace(num_envs=3, device="cpu", step_dt=0.02,
                          episode_length_buf=torch.tensor([0, 50, 200]),
                          _headstand_spawn_inverted=torch.tensor([-0.24, -0.24, -0.24]))
    sp = microduck_mdp._headstand_setpoint(env)
    assert abs(float(sp[0]) + 0.24) < 1e-6          # at t=0 the setpoint is the pike's inversion
    assert -0.24 < float(sp[1]) < 1.0                # halfway through the ramp
    assert abs(float(sp[2]) - 1.0) < 1e-6            # after 2 s it has arrived
    microduck_mdp._HEADSTAND_RAMP_S = 0.0
    assert torch.allclose(microduck_mdp._headstand_setpoint(env), torch.ones(3))


def test_backroll_starts_in_the_hold_and_never_standing():
    from mjlab_microduck.tasks.microduck_backroll_env_cfg import make_microduck_backroll_env_cfg
    cfg = make_microduck_backroll_env_cfg(style="straight")
    p = cfg.events["set_roulade_state"].params
    assert p["standing_prob"] == 0.0 and p["midroll_prob"] == 1.0
    assert math.radians(160) < p["midroll_pitch_min"] < p["midroll_pitch_max"] < math.radians(195)
    assert p["tuck_factor_range"][1] == 1.0
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
    assert cfg.scene.entities["robot"] is MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
    for stage in cfg.curriculum["roulade_spawn_mix"].params["param_stages"]:
        assert stage["params"]["standing_prob"] == 0.0


def test_split_exit_spawns_in_the_hold_and_targets_the_pike():
    cfg = make_microduck_headstand_env_cfg(style="splitexit")
    p = cfg.events["set_headstand_spawn"].params
    assert p["hold_prob"] == 1.0 and p["standing_prob"] == 0.0 and p["tripod_prob"] == 0.0
    assert cfg.rewards["fold_progress"].func is microduck_mdp.fold_progress_down
    assert "fold_composite" in cfg.rewards and "fold_overshoot" not in cfg.rewards
    assert "flopped" not in cfg.terminations


def test_split_exit_frontier_ignores_a_backward_fall():
    import torch
    from types import SimpleNamespace
    # Trunk quaternions: inverted (pitch 180) then fallen backward (pitch 250 = -110 wrapped).
    def q(deg):
        p = math.radians(deg); return [math.cos(p / 2), 0.0, math.sin(p / 2), 0.0]
    pitch = microduck_mdp._trunk_pitch(SimpleNamespace(data=SimpleNamespace(root_link_quat_w=torch.tensor([q(180), q(250)]))))
    unwrapped = torch.where(pitch < 0, pitch + 2 * math.pi, pitch)
    assert unwrapped[1] > unwrapped[0] > math.radians(170)   # backward = the angle keeps growing, never "down to the pike"


def test_split_switch_flag_and_mirror():
    cfg = make_microduck_headstand_env_cfg(kickup=True, style="split", switch=True)
    assert isinstance(cfg.commands["twist"], microduck_mdp.SplitSwitchCommandCfg)
    assert cfg.rewards["headstand_composite"].params["switch_command"] == "twist"
    from mjlab_microduck.tasks.microduck_headstand_env_cfg import HEADSTAND_OVERRIDES
    m = microduck_mdp._mirror_overrides(HEADSTAND_OVERRIDES)
    assert m[2] == -HEADSTAND_OVERRIDES[11] and m[11] == -HEADSTAND_OVERRIDES[2]   # legs swapped, signs flipped
    assert m[5] == HEADSTAND_OVERRIDES[5] and m[6] == HEADSTAND_OVERRIDES[6]         # neck untouched
    assert microduck_mdp._mirror_overrides(m) == HEADSTAND_OVERRIDES                 # an involution
