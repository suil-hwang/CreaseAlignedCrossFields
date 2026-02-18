from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense, Var

_MINIMIZE = ObjectiveSense.Minimize  # type: ignore[attr-defined]


class MosekSoftCrossFieldsError(RuntimeError):
    pass


def _as_float_vector(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _to_fusion_matrix(A: sp.spmatrix | np.ndarray) -> Matrix:
    if sp.issparse(A):
        coo = sp.coo_matrix(A)
        return Matrix.sparse(
            int(coo.shape[0]),
            int(coo.shape[1]),
            coo.row.astype(np.int32, copy=False),
            coo.col.astype(np.int32, copy=False),
            coo.data.astype(np.float64, copy=False),
        )

    arr = np.asarray(A, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Dense matrices must be 2D.")
    return Matrix.dense(arr)


def mosek_soft_cross_fields_wrapper(
    A: sp.spmatrix | np.ndarray,
    b: np.ndarray,
    D: sp.spmatrix | np.ndarray,
    a: np.ndarray,
    psymbol: float,
    n: float,
    c: np.ndarray,
    *,
    verbose: bool = False,
    x0: np.ndarray | None = None,
    solver_opts: dict[str, Any] | None = None,
) -> np.ndarray:
    A_mat = A if sp.issparse(A) else np.asarray(A, dtype=np.float64)
    D_mat = D if sp.issparse(D) else np.asarray(D, dtype=np.float64)

    b_vec = _as_float_vector("b", b)
    a_vec = _as_float_vector("a", a)
    c_vec = _as_float_vector("c", c)

    if A_mat.shape[0] != b_vec.size:
        raise ValueError("A and b dimensions are incompatible.")
    if D_mat.shape[0] != c_vec.size:
        raise ValueError("D and c dimensions are incompatible.")
    if D_mat.shape[1] != A_mat.shape[1]:
        raise ValueError("A and D must have the same number of columns.")

    xdim = int(A_mat.shape[1])
    if xdim % 9 != 0:
        raise ValueError("Unknown variable dimension; expected multiples of 9.")
    nfaces = xdim // 9
    if nfaces == 0:
        return np.zeros(0, dtype=np.float64)

    if b_vec.size % nfaces != 0:
        raise ValueError("b size is not compatible with the number of faces.")
    rows_per_face = b_vec.size // nfaces
    if rows_per_face != 7:
        raise ValueError("Expected 7 normal-alignment rows per face.")

    if c_vec.size % 9 != 0:
        raise ValueError("c must encode 9D edge vectors.")
    nedges = c_vec.size // 9
    if a_vec.size != nedges:
        raise ValueError("a must have one entry per edge.")

    actual_p = np.inf if psymbol == -1 else float(psymbol)
    if actual_p < 1.0:
        raise ValueError("p must be >= 1, or -1 for infinity norm.")
    if float(n) < 0.0:
        raise ValueError("n must be nonnegative.")

    Af = _to_fusion_matrix(A_mat)
    Df = _to_fusion_matrix(D_mat)
    x0_vec: np.ndarray | None = None
    if x0 is not None:
        x0_vec = _as_float_vector("x0", x0)
        if x0_vec.size != xdim:
            raise ValueError("x0 has invalid length.")

    try:
        with Model("MosekSoftCrossFields") as M:
            M.setSolverParam("intpntSolveForm", "dual")
            M.setSolverParam("numThreads", 0)
            if solver_opts:
                for key, value in solver_opts.items():
                    M.setSolverParam(str(key), value)

            if verbose:
                import sys

                M.setLogHandler(sys.stdout)

            x = M.variable("x", xdim, Domain.unbounded())
            y = M.variable("y", 9 * nedges, Domain.unbounded())
            ynorms = M.variable("ynorms", nedges, Domain.greaterThan(0.0))
            z = M.variable("z", 1, Domain.greaterThan(0.0))
            if x0_vec is not None:
                x.setLevel(x0_vec)

            M.objective(_MINIMIZE, z)

            Axmb = Expr.sub(Expr.mul(Af, x), Expr.constTerm(b_vec))
            if float(n) == 0.0:
                M.constraint("normal_alignment_eq", Axmb, Domain.equalsTo(0.0))
            else:
                Axmb_reshaped = Expr.reshape(Axmb, nfaces, rows_per_face)
                n_repeated = Expr.repeat(Expr.constTerm(float(n)), nfaces, 0)
                M.constraint(
                    "normal_alignment_qcone",
                    Expr.hstack(n_repeated, Axmb_reshaped),
                    Domain.inQCone(nfaces, rows_per_face + 1),
                )

            M.constraint(
                "define_y",
                Expr.sub(y, Expr.add(Expr.mul(Df, x), Expr.constTerm(c_vec))),
                Domain.equalsTo(0.0),
            )

            y_reshaped = Expr.reshape(y, nedges, 9)
            M.constraint("y_norm_qcone", Expr.hstack(ynorms, y_reshaped), Domain.inQCone(nedges, 10))

            if actual_p == 1.0:
                M.constraint("p_norm_l1", Expr.sub(z, Expr.dot(a_vec, ynorms)), Domain.greaterThan(0.0))
            elif actual_p == 2.0:
                sqrta = np.sqrt(a_vec)
                M.constraint("p_norm_l2", Expr.vstack(z, Expr.mulElm(sqrta, ynorms)), Domain.inQCone())
            elif np.isinf(actual_p):
                M.constraint("p_norm_linf", Expr.sub(Var.repeat(z, nedges), ynorms), Domain.greaterThan(0.0))
            else:
                alpha = 1.0 - 1.0 / float(actual_p)
                powa = np.power(a_vec, 1.0 / float(actual_p))
                r = M.variable("r", nedges)
                M.constraint("p_norm_sum", Expr.sub(z, Expr.sum(r)), Domain.equalsTo(0.0))
                M.constraint(
                    "p_norm_powercone",
                    Expr.hstack(Var.repeat(z, nedges), r, Expr.mulElm(powa, ynorms)),
                    Domain.inPPowerCone(alpha, nedges),
                )

            M.solve()
            return np.asarray(x.level(), dtype=np.float64).reshape(-1)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise MosekSoftCrossFieldsError(f"MOSEK Fusion solve failed: {exc}") from exc
