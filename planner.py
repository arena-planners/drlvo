"""DRL-VO wrapper for the arena_planners bridge."""

from __future__ import annotations

import os
import pathlib

import numpy as np
from arena_planners.sdk import load_manifest, main_loop

from policy import DrlVoPolicy

_WEIGHTS = os.path.join(os.path.dirname(__file__), "model", "drl_vo")

_RANGE_LIMIT: float = 30.0
_VX_LIMIT: float = 0.5
_WZ_LIMIT: float = 0.7
_SCAN_BEAMS: int = 720
_SCAN_HISTORY_LEN: int = 10
_GOAL_MAX_DIST: float = 2.0
_ARRIVE_R: float = 1.0
_STOP_R: float = 0.2
_PLAN_END_EPS: float = 0.05
_PED_CELL: float = 0.25


def ped_velocity_map(
    peds: np.ndarray | None, robot_pose: np.ndarray | None
) -> np.ndarray:
    """(2, 80, 80) robot-frame pedestrian velocity map, built as DRL-VO's cnn_data_pub.ped_callback."""
    m = np.zeros((2, 80, 80), dtype=np.float32)
    if peds is None or len(peds) == 0 or robot_pose is None or len(robot_pose) < 3:
        return m
    rx, ry, th = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    c_, s_ = np.cos(-th), np.sin(-th)
    for _, px, py, vx, vy in np.asarray(peds, dtype=np.float64)[:, :5]:
        dx, dy = px - rx, py - ry
        x, y = c_ * dx - s_ * dy, s_ * dx + c_ * dy
        if 0.0 <= x <= 20.0 and abs(y) <= 10.0:
            r = min(int(np.floor(x / _PED_CELL)), 79)
            c = min(int(np.floor(-(y - 10.0) / _PED_CELL)), 79)
            m[0, r, c] = c_ * vx - s_ * vy
            m[1, r, c] = s_ * vx + c_ * vy
    return m


_policy: DrlVoPolicy | None = None


def _get_policy() -> DrlVoPolicy:
    global _policy
    if _policy is None:
        _policy = DrlVoPolicy(
            weights_path=_WEIGHTS,
            range_limit=_RANGE_LIMIT,
            vx_limit=_VX_LIMIT,
            wz_limit=_WZ_LIMIT,
        )
    return _policy


_scan_history_buffer: list[list[float]] = []


def _at_plan_end(goal_pose: np.ndarray, plan: np.ndarray | None) -> bool:
    """True when the subgoal is the last plan point, or no plan exists and the subgoal is the goal itself."""
    if plan is None or len(plan) == 0:
        return True
    return (
        float(np.hypot(goal_pose[0] - plan[-1][0], goal_pose[1] - plan[-1][1]))
        < _PLAN_END_EPS
    )


def step(features: dict) -> list[float]:
    """Compute [v, omega] from raw obs (robot_pose, goal_pose, laser_scan, pedestrians)."""
    policy = _get_policy()

    scan_raw = features.get("laser_scan")
    if scan_raw is not None and len(scan_raw) > 0:
        # canonical scan is heading-first, the policy expects the heading at len/2
        canonical = np.asarray(scan_raw, dtype=np.float32)
        current_scan = np.roll(canonical, canonical.size // 2).tolist()
    else:
        current_scan = [_RANGE_LIMIT] * _SCAN_BEAMS

    _scan_history_buffer.append(current_scan)
    if len(_scan_history_buffer) > _SCAN_HISTORY_LEN:
        del _scan_history_buffer[0]
    while len(_scan_history_buffer) < _SCAN_HISTORY_LEN:
        _scan_history_buffer.insert(0, current_scan)
    scan_history = [v for frame in _scan_history_buffer for v in frame]

    robot_pose = features.get("robot_pose")
    goal_pose = features.get("goal_pose")

    sub_goal_rf: tuple[float, float] | None = None
    if robot_pose is not None and goal_pose is not None and len(robot_pose) >= 3 and len(goal_pose) >= 2:
        dx = float(goal_pose[0] - robot_pose[0])
        dy = float(goal_pose[1] - robot_pose[1])
        theta = float(robot_pose[2])
        dist = float(np.hypot(dx, dy))
        angle = (np.arctan2(dy, dx) - theta + np.pi) % (2 * np.pi) - np.pi
        sub_goal_rf = (
            float(np.cos(angle) * min(dist, _GOAL_MAX_DIST)),
            float(np.sin(angle) * min(dist, _GOAL_MAX_DIST)),
        )

    sub_goal = [sub_goal_rf[0], sub_goal_rf[1]] if sub_goal_rf is not None else [0.0, 0.0]

    if (
        sub_goal_rf is not None
        and dist <= _ARRIVE_R
        and _at_plan_end(goal_pose, features.get("global_plan"))
    ):
        if dist <= _STOP_R:
            return [0.0, 0.0]
        wz = float(np.clip(1.5 * angle, -_WZ_LIMIT, _WZ_LIMIT))
        vx = float(np.clip(0.5 * dist, 0.0, 0.3) * max(0.0, np.cos(angle)))
        n = len(current_scan)
        if (
            min(current_scan[int(n / 2 - n / 9) : int(n / 2 + n / 9)] or [_RANGE_LIMIT])
            <= 0.45
        ):
            vx = 0.0
        return [vx, wz]

    vx, wz = policy.compute_velocity_commands(
        scan_history=scan_history,
        current_scan=current_scan,
        sub_goal=sub_goal,
        ped_map=ped_velocity_map(features.get("pedestrians"), robot_pose),
    )
    return [vx, wz]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    global _policy
    _policy = None
    _scan_history_buffer.clear()


if __name__ == "__main__":
    manifest = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
    main_loop(step, manifest=manifest)
