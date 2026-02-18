from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

_ROOT = Path(__file__).resolve().parent

from arff_io import coeff2_frames, frames2_octa
from mbo import OctaMBO
from variety import octa_align_mat

from getMeshData import get_mesh_data
from MosekSoftCrossFields.MosekSoftCrossFieldsWrapper import mosek_soft_cross_fields_wrapper
from rank3tensor2blockdiag import rank3tensor2blockdiag


def _flat_block_indices(idx: np.ndarray, block: int = 9) -> np.ndarray:
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    return (idx[:, None] * block + np.arange(block, dtype=np.int64)[None, :]).reshape(-1)


def _project_bad_frames_with_retries(
    fiber: Any,
    frames_unproj: np.ndarray,
    bad_frame_inds: np.ndarray,
    *,
    max_retries: int | None,
    base_perturbation: float,
    perturbation_growth: float,
) -> np.ndarray:
    if bad_frame_inds.size == 0:
        return np.zeros((0, 9), dtype=np.float64)

    bad_unproj_t = frames_unproj[bad_frame_inds, :].T
    randpert = np.zeros_like(bad_unproj_t)
    attempts = 0
    scale = float(base_perturbation)

    while True:
        try:
            return fiber.proj(bad_unproj_t + randpert).T
        except Exception as exc:
            attempts += 1
            if max_retries is not None and attempts >= max_retries:
                raise RuntimeError(
                    f"Frame projection failed after {attempts} attempts."
                ) from exc

            randpert = (np.random.random(bad_unproj_t.shape) - 0.5) * scale
            scale *= float(perturbation_growth)


def _resolve_visualization_length(
    vertices: np.ndarray,
    *,
    mode: str,
    scale: float,
    fixed_length: float | None,
) -> float:
    if fixed_length is not None:
        return float(max(fixed_length, 0.0))

    vmin = np.min(vertices, axis=0)
    vmax = np.max(vertices, axis=0)
    diag = float(np.linalg.norm(vmax - vmin))
    if diag <= 0.0:
        diag = 1.0

    mode_l = mode.lower()
    if mode_l == "bbox":
        return float(diag * scale)
    if mode_l == "unit":
        return float(scale)
    raise ValueError("visualization_length_mode must be 'bbox' or 'unit'.")


def solve_lp_cross_field(
    X: np.ndarray,
    T: np.ndarray,
    mname: str = "",
    n: float = 0.0,
    p: float = 2.0,
    Visualize: bool = False,
    isFixedTriangle: np.ndarray | None = None,
    fixedTriangleFrames: np.ndarray | None = None,
    *,
    max_meta_iterations: int = 10,
    projection_threshold: float = 0.665,
    max_projection_retries: int | None = 200,
    projection_perturbation: float = 1e-12,
    projection_perturbation_growth: float = 2.0,
    visualization_length_mode: str = "bbox",
    visualization_length_scale: float = 0.01,
    visualization_length: float | None = None,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    if p < 1 and not np.isinf(p):
        raise ValueError("p must be >= 1 (or np.inf).")

    if isFixedTriangle is None or np.asarray(isFixedTriangle).size == 0:
        is_fixed = None
        fixed_frames = None
    else:
        is_fixed = np.asarray(isFixedTriangle, dtype=bool).reshape(-1)
        fixed_frames = np.asarray(fixedTriangleFrames, dtype=np.float64)
        if fixed_frames.ndim != 3 or fixed_frames.shape[0] != 3 or fixed_frames.shape[1] != 3:
            raise ValueError("fixedTriangleFrames must have shape (3, 3, k).")
        if int(is_fixed.sum()) != fixed_frames.shape[2]:
            raise ValueError("sum(isFixedTriangle) must match fixedTriangleFrames.shape[2].")

    should_save = len(mname) > 0
    fname = ""
    if should_save:
        outdir = _ROOT / "Results"
        outdir.mkdir(parents=True, exist_ok=True)
        if is_fixed is not None:
            token = int(np.random.randint(0, 1000))
            fname = str(outdir / f"mesh_{mname}_n_{n:g}_p_{p:g}_dc_{token}.mat")
        else:
            fname = str(outdir / f"mesh_{mname}_n_{n:g}_p_{p:g}.mat")

    if fname and Path(fname).exists():
        return np.zeros((3, 3, 0), dtype=np.float64), "Skipped", {}

    fiber = OctaMBO()
    data = get_mesh_data(X, T)

    num_tri = int(data["numTriangles"])

    u0_one = np.array([0.0, 0.0, 0.0, np.sqrt(7.0 / 12.0), 0.0, 0.0, 0.0], dtype=np.float64)
    u0 = np.tile(u0_one, num_tri)

    D = np.asarray(octa_align_mat(data["faceNormals"]), dtype=np.float64)
    D[np.abs(D) <= 1e-6] = 0.0

    N7 = rank3tensor2blockdiag(D[1:8, :, :])

    prepocW = np.asarray(data["primalOverDualWeight"], dtype=np.float64)

    bad_frame_inds = np.arange(num_tri, dtype=np.int64)
    good_frame_inds = np.zeros(0, dtype=np.int64)
    frames_proj = np.zeros((num_tri, 9), dtype=np.float64)

    if is_fixed is not None:
        assert fixed_frames is not None
        bad_frame_inds = np.where(~is_fixed)[0].astype(np.int64)
        good_frame_inds = np.where(is_fixed)[0].astype(np.int64)
        frames_proj[good_frame_inds, :] = frames2_octa(fixed_frames).T

    psymbol = -1 if np.isinf(p) else float(p)
    counter = 0
    mosek_error = False
    bad_frames_per_meta_iter: list[int] = []
    good_frames_per_meta_iter: list[int] = []
    frames: list[dict[str, Any]] = []

    DM = cast(sp.csr_matrix, sp.kron(data["incidenceMatrix"], sp.eye(9, dtype=np.float64), format="csr"))
    num_dual_rows = int(DM.shape[0])

    x = frames_proj.T.reshape(-1, order="F")
    dirs1 = np.zeros((3, 3, num_tri), dtype=np.float64)

    while bad_frame_inds.size > 0 and counter < int(max_meta_iterations) and not mosek_error:
        t1 = time.perf_counter()
        counter += 1

        good_flat = _flat_block_indices(good_frame_inds, block=9)
        bad_flat = _flat_block_indices(bad_frame_inds, block=9)

        AM = rank3tensor2blockdiag(D[1:8, :, bad_frame_inds])
        known = frames_proj.T.reshape(-1, order="F")

        DM_bad = cast(sp.csr_matrix, DM[:, bad_flat])
        if good_flat.size > 0:
            DM_good = cast(sp.csr_matrix, DM[:, good_flat])
            cm = np.asarray(DM_good @ known[good_flat], dtype=np.float64).reshape(-1)
        else:
            cm = np.zeros(num_dual_rows, dtype=np.float64)

        try:
            xbad = mosek_soft_cross_fields_wrapper(
                AM,
                u0[: 7 * bad_frame_inds.size],
                DM_bad,
                prepocW,
                psymbol,
                float(n),
                cm,
                x0=x[bad_flat],
            )
        except Exception as exc:
            mosek_error = True
            if "Unknown" in str(exc) and bad_flat.size <= x.size:
                xbad = x[bad_flat].copy()
            else:
                raise
        x[bad_flat] = xbad

        seconds_elapsed = time.perf_counter() - t1

        frames_unproj_raw = x.reshape((9, num_tri), order="F").T
        row_norm = np.linalg.norm(frames_unproj_raw, axis=1, keepdims=True)
        frames_unproj = np.divide(
            frames_unproj_raw,
            row_norm,
            out=np.zeros_like(frames_unproj_raw),
            where=row_norm > 0.0,
        )

        frames_proj[bad_frame_inds, :] = _project_bad_frames_with_retries(
            fiber,
            frames_unproj,
            bad_frame_inds,
            max_retries=max_projection_retries,
            base_perturbation=projection_perturbation,
            perturbation_growth=projection_perturbation_growth,
        )

        deviation = np.linalg.norm(frames_unproj - frames_proj, axis=1)
        good_frame_inds = np.where(deviation < float(projection_threshold))[0].astype(np.int64)
        bad_frame_inds = np.where(deviation >= float(projection_threshold))[0].astype(np.int64)

        if counter > 1 and bad_frame_inds.size == bad_frames_per_meta_iter[-1]:
            break

        dirs1 = coeff2_frames(fiber.proj(frames_proj.T))
        dirs2 = np.concatenate([dirs1, -dirs1], axis=1).transpose(2, 1, 0).reshape(-1, 3)

        residual = np.asarray(N7 @ x - u0, dtype=np.float64)
        alignment_colors = np.linalg.norm(residual.reshape((7, num_tri), order="F").T, axis=1)

        good_frames_per_meta_iter.append(int(good_frame_inds.size))
        bad_frames_per_meta_iter.append(int(bad_frame_inds.size))
        frames.append(
            {
                "secondsElapsed": float(seconds_elapsed),
                "frames": frames_proj.copy(),
                "framesUnprojRaw": frames_unproj_raw.copy(),
                "goodFrameInds": good_frame_inds.copy(),
                "badFrameInds": bad_frame_inds.copy(),
                "alignmentColors": alignment_colors,
                "dirs1": dirs1.copy(),
                "dirs2": dirs2.copy(),
            }
        )

    need_lines = Visualize or (should_save and fname)
    lines = (
        np.repeat(np.asarray(data["triangleBarycenters"], dtype=np.float64), 6, axis=0)
        if need_lines
        else np.empty((0, 3), dtype=np.float64)
    )

    if Visualize:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        dirs1v = coeff2_frames(frames_proj.T)
        dirs2v = np.concatenate([dirs1v, -dirs1v], axis=1).transpose(2, 1, 0).reshape(-1, 3)
        quiver_length = _resolve_visualization_length(
            np.asarray(data["vertices"], dtype=np.float64),
            mode=visualization_length_mode,
            scale=float(visualization_length_scale),
            fixed_length=visualization_length,
        )

        fig = plt.figure()
        ax = cast(Any, fig.add_subplot(111, projection="3d"))

        tri = data["triangles"]
        verts = data["vertices"]
        poly = Poly3DCollection(verts[tri], alpha=0.35, facecolor="green", edgecolor="k", linewidth=0.1)
        ax.add_collection3d(poly)

        ax.quiver(
            lines[:, 0],
            lines[:, 1],
            lines[:, 2],
            dirs2v[:, 0],
            dirs2v[:, 1],
            dirs2v[:, 2],
            color="b",
            length=quiver_length,
            normalize=False,
        )
        ax.set_box_aspect((1.0, 1.0, 1.0))
        plt.show()

    if should_save and fname:
        max_proj_retries_save = -1 if max_projection_retries is None else int(max_projection_retries)
        to_save: dict[str, Any] = {
            "frames": np.array(frames, dtype=object),
            "mosekError": mosek_error,
            "mname": mname,
            "metaIterationCounter": counter,
            "goodFramesPerMetaIter": np.asarray(good_frames_per_meta_iter, dtype=np.int64),
            "badFramesPerMetaIter": np.asarray(bad_frames_per_meta_iter, dtype=np.int64),
            "X": data["vertices"],
            "T": data["triangles"],
            "data": data,
            "lines": lines,
            "n": float(n),
            "p": float(p),
            "solver": "MOSEK",
            "projectionThreshold": float(projection_threshold),
            "maxMetaIterations": int(max_meta_iterations),
            "maxProjectionRetries": max_proj_retries_save,
            "projectionPerturbation": float(projection_perturbation),
            "projectionPerturbationGrowth": float(projection_perturbation_growth),
            "visualizationLengthMode": str(visualization_length_mode),
            "visualizationLengthScale": float(visualization_length_scale),
            "visualizationLength": (
                float(visualization_length)
                if visualization_length is not None
                else np.nan
            ),
            "finalFramesProj": frames_proj.copy(),
            "finalDirs1": dirs1.copy(),
        }
        sio.savemat(fname, {"toSave": to_save}, do_compression=True)

    return dirs1, fname, data


def SolveLpCrossField(
    X: np.ndarray,
    T: np.ndarray,
    mname: str,
    n: float,
    p: float,
    Visualize: bool,
    isFixedTriangle: np.ndarray | None = None,
    fixedTriangleFrames: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    return solve_lp_cross_field(
        X,
        T,
        mname,
        n,
        p,
        Visualize,
        isFixedTriangle,
        fixedTriangleFrames,
        **kwargs,
    )
