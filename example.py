from __future__ import annotations

import argparse
from pathlib import Path

import meshio
import numpy as np

from SolveLpCrossField import SolveLpCrossField


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run crease-aligned cross-field example.")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path("Meshes") / "notch5.obj",
        help="Path to input triangle mesh file.",
    )
    parser.add_argument(
        "--mname",
        type=str,
        default=None,
        help="Result name prefix. Defaults to mesh filename stem.",
    )
    parser.add_argument(
        "--normal-alignment",
        type=float,
        default=0.0,
        help="Normal alignment softness (n).",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=2.0,
        help="Objective p-norm.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show final field visualization.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    root = Path(__file__).resolve().parent
    fname = args.mesh if args.mesh.is_absolute() else (root / args.mesh)

    mname = args.mname if args.mname is not None else fname.stem
    normal_alignment = float(args.normal_alignment)
    pnorm = float(args.p)
    should_visualize = bool(args.visualize)

    mesh = meshio.read(fname)
    X = np.asarray(mesh.points, dtype=np.float64)
    T = np.asarray(mesh.cells_dict["triangle"], dtype=np.int64)

    is_fixed_triangle = None
    fixed_triangle_frames = None

    dirs1, saved_name, _ = SolveLpCrossField(
        X,
        T,
        mname,
        normal_alignment,
        pnorm,
        should_visualize,
        is_fixed_triangle,
        fixed_triangle_frames,
    )

    print(f"dirs1 shape: {dirs1.shape}")
    print(f"saved: {saved_name}")


if __name__ == "__main__":
    main()

# Usage:
# python example.py --mesh Meshes/twistcube90.obj