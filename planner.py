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


def step(features: dict) -> list[float]:
    """Compute [v, omega] from raw obs (robot_pose, goal_pose, laser_scan, pedestrians)."""
    policy = _get_policy()

    scan_raw = features.get("laser_scan")
    if scan_raw is not None and len(scan_raw) > 0:
        current_scan = np.asarray(scan_raw, dtype=np.float32).tolist()
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

    vx, wz = policy.compute_velocity_commands(
        scan_history=scan_history,
        current_scan=current_scan,
        sub_goal=sub_goal,
    )
    return [vx, wz]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    global _policy
    _policy = None
    _scan_history_buffer.clear()


if __name__ == "__main__":
    manifest = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
    main_loop(step, manifest=manifest)
