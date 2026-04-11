#!/usr/bin/env python3
"""Hand-eye calibration utility for camera-to-gripper extrinsics.

This script expects image filenames to include six robot joint values in brackets:
    [q1, q2, q3, q4, q5, q6].png

Units:
- Kinematics and calibration are computed in meters and radians.
- Final results are printed in both meters and millimeters.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# MyCobot280 DH parameters:
# (theta_offset_rad, d_m, a_m, alpha_rad)
DH_PARAMS: list[tuple[float, float, float, float]] = [
    (0.0, 131.22 / 1000.0, 0.0 / 1000.0, 1.5708),
    (-1.5708, 0.0 / 1000.0, -110.4 / 1000.0, 0.0),
    (0.0, 0.0 / 1000.0, -96.0 / 1000.0, 0.0),
    (-1.5708, 63.4 / 1000.0, 0.0 / 1000.0, 1.5708),
    (1.5708, 75.05 / 1000.0, 0.0 / 1000.0, -1.5708),
    (0.0, 45.6 / 1000.0, 0.0 / 1000.0, 0.0),
]

CAMERA_MATRIX = np.array(
    [
        [986.93180332, 0.0, 251.57937661],
        [0.0, 983.85767546, 131.93997962],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DIST_COEFFS = np.array(
    [-2.77060722e-01, -1.73446994e00, 4.61497572e-03, 4.60231034e-03, 6.43334573e00],
    dtype=np.float64,
)

JOINT_PATTERN = re.compile(r"\[([^\]]+)\]")


@dataclass
class CalibrationConfig:
    image_dir: Path
    pattern_size: tuple[int, int]  # (columns, rows) of inner corners
    square_size_m: float
    joint_unit: str  # "radian" or "degree"
    method: int


def fk_from_dh(joints_rad: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Compute base-to-gripper pose from 6-DoF joints (radians)."""
    joints = list(joints_rad)
    if len(joints) != 6:
        raise ValueError(f"Expected 6 joints, got {len(joints)}")

    transform = np.eye(4, dtype=np.float64)

    for (theta_offset, d_m, a_m, alpha), theta in zip(DH_PARAMS, joints):
        total_theta = theta + theta_offset
        ct, st = np.cos(total_theta), np.sin(total_theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        a_i = np.array(
            [
                [ct, -st * ca, st * sa, a_m * ct],
                [st, ct * ca, -ct * sa, a_m * st],
                [0.0, sa, ca, d_m],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        transform = transform @ a_i

    rot = transform[:3, :3].copy()
    trans = transform[:3, 3].reshape(3, 1).copy()
    return rot, trans


def parse_joints_from_filename(filename: str, joint_unit: str) -> list[float]:
    """Extract six joint values from filename and convert to radians if needed."""
    stem = Path(filename).stem
    match = JOINT_PATTERN.search(stem)
    if not match:
        raise ValueError(f"No bracketed joint values found: {filename}")

    parts = [chunk.strip() for chunk in match.group(1).split(",")]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 joint values in filename, got {len(parts)}: {filename}")

    joints = [float(value) for value in parts]

    if joint_unit == "radian":
        return joints
    if joint_unit == "degree":
        return [float(np.deg2rad(v)) for v in joints]
    raise ValueError("joint_unit must be 'radian' or 'degree'")


def invert_pose(rot: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert rigid transform represented by (R, t)."""
    r_inv = rot.T
    t_inv = -r_inv @ trans
    return r_inv, t_inv


def build_object_points(pattern_size: tuple[int, int], square_size_m: float) -> np.ndarray:
    """Create chessboard object points in board coordinates."""
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.indices((cols, rows)).T.reshape(-1, 2)
    objp[:, :2] = grid * square_size_m
    return objp


def collect_data(config: CalibrationConfig) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Collect hand-eye calibration samples from images and joint-encoded filenames."""
    if not config.image_dir.exists() or not config.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {config.image_dir}")

    objp = build_object_points(config.pattern_size, config.square_size_m)

    r_gripper2base: list[np.ndarray] = []
    t_gripper2base: list[np.ndarray] = []
    r_target2cam: list[np.ndarray] = []
    t_target2cam: list[np.ndarray] = []

    image_files = sorted(
        p for p in config.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if not image_files:
        raise RuntimeError(f"No images found in {config.image_dir}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    used = 0
    skipped = 0

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[skip] Failed to load image: {img_path.name}")
            skipped += 1
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, config.pattern_size, None)

        if not found:
            print(f"[skip] Chessboard not found: {img_path.name}")
            skipped += 1
            continue

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ok, rvec, tvec = cv2.solvePnP(objp, corners, CAMERA_MATRIX, DIST_COEFFS)
        if not ok:
            print(f"[skip] solvePnP failed: {img_path.name}")
            skipped += 1
            continue

        joints_rad = parse_joints_from_filename(img_path.name, config.joint_unit)
        r_base2gripper, t_base2gripper = fk_from_dh(joints_rad)
        r_g2b, t_g2b = invert_pose(r_base2gripper, t_base2gripper)

        r_t2c, _ = cv2.Rodrigues(rvec)
        t_t2c = tvec.reshape(3, 1)

        r_gripper2base.append(r_g2b)
        t_gripper2base.append(t_g2b)
        r_target2cam.append(r_t2c)
        t_target2cam.append(t_t2c)
        used += 1

        print(f"[ok] {img_path.name} | joints(rad)={np.round(joints_rad, 4).tolist()}")

    print(f"Samples used: {used}, skipped: {skipped}, total: {len(image_files)}")

    if used < 3:
        raise RuntimeError("At least 3 valid samples are required for hand-eye calibration")

    return r_gripper2base, t_gripper2base, r_target2cam, t_target2cam


def method_from_name(name: str) -> int:
    table = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    lower = name.lower()
    if lower not in table:
        raise ValueError(f"Unsupported hand-eye method: {name}")
    return table[lower]


def parse_args() -> CalibrationConfig:
    parser = argparse.ArgumentParser(description="Hand-eye calibration runner")
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Folder containing calibration images with joint values in filenames",
    )
    parser.add_argument("--pattern-cols", type=int, default=6, help="Chessboard inner corners (columns)")
    parser.add_argument("--pattern-rows", type=int, default=9, help="Chessboard inner corners (rows)")
    parser.add_argument("--square-size-m", type=float, default=0.020, help="Square size in meters")
    parser.add_argument(
        "--joint-unit",
        choices=["radian", "degree"],
        default="radian",
        help="Unit for joint values embedded in image filenames",
    )
    parser.add_argument(
        "--method",
        default="park",
        help="calibrateHandEye method: tsai, park, horaud, andreff, daniilidis",
    )
    args = parser.parse_args()

    return CalibrationConfig(
        image_dir=args.image_dir,
        pattern_size=(args.pattern_cols, args.pattern_rows),
        square_size_m=args.square_size_m,
        joint_unit=args.joint_unit,
        method=method_from_name(args.method),
    )


def main() -> None:
    config = parse_args()

    r_g2b, t_g2b, r_t2c, t_t2c = collect_data(config)

    r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        r_g2b,
        t_g2b,
        r_t2c,
        t_t2c,
        method=config.method,
    )

    t_cam2gripper_mm = t_cam2gripper * 1000.0

    transform_m = np.eye(4, dtype=np.float64)
    transform_m[:3, :3] = r_cam2gripper
    transform_m[:3, 3] = t_cam2gripper.reshape(3)

    transform_mm = np.eye(4, dtype=np.float64)
    transform_mm[:3, :3] = r_cam2gripper
    transform_mm[:3, 3] = t_cam2gripper_mm.reshape(3)

    np.set_printoptions(precision=6, suppress=True)

    print("\n=== Hand-Eye Result (m) ===")
    print("R_cam2gripper:\n", r_cam2gripper)
    print("t_cam2gripper (m):\n", t_cam2gripper.reshape(3))
    print("T_cam2gripper (4x4, m):\n", transform_m)

    print("\n=== Hand-Eye Result (mm) ===")
    print("t_cam2gripper (mm):\n", t_cam2gripper_mm.reshape(3))
    print("T_cam2gripper (4x4, mm):\n", transform_mm)


if __name__ == "__main__":
    main()
