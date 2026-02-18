from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from MosekSoftCrossFields.MosekSoftCrossFieldsWrapper import mosek_soft_cross_fields_wrapper
from rank3tensor2blockdiag import rank3tensor2blockdiag


def mosek_soft_cross_fields_test(
    A: sp.spmatrix | np.ndarray | None = None,
    b: np.ndarray | None = None,
    D: sp.spmatrix | np.ndarray | None = None,
    a: np.ndarray | None = None,
    p: float | None = None,
    n: float | None = None,
    *,
    seed: int | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    if A is None:
        nedges = int(rng.integers(3, 13))
        nfaces = int(rng.integers(3, 13))
        dim_per_edge = 9
        D = rng.random((dim_per_edge * nedges, dim_per_edge * nfaces)) - 0.5
        A = rank3tensor2blockdiag(rng.random((dim_per_edge - 2, dim_per_edge, nfaces)) - 0.5)
        psymbol = float(rng.choice(np.array([-1, 1, 2, 10, 30, 50], dtype=np.float64)))
        a = rng.random(nedges)
        b = rng.random((dim_per_edge - 2) * nfaces) - 0.5
        n = float(rng.choice(np.linspace(0.0, 1.0, 11)))
        c = rng.random(nedges * dim_per_edge) - 0.5
    else:
        assert b is not None and D is not None and a is not None and p is not None and n is not None
        psymbol = -1 if p == np.inf else float(p)
        c = np.zeros(np.asarray(D).shape[0], dtype=np.float64)

    xmosek = mosek_soft_cross_fields_wrapper(A, b, D, a, psymbol, float(n), c)

    Dmat = D if sp.issparse(D) else np.asarray(D, dtype=np.float64)
    actual_p = np.inf if psymbol == -1 else float(psymbol)
    ap = np.power(a, 1.0 / actual_p) if np.isfinite(actual_p) else np.ones_like(a)
    y = np.reshape(Dmat @ xmosek + c, (9, -1), order="F").T
    edge_norms = np.linalg.norm(y, axis=1) * ap
    energy = float(np.max(edge_norms) if np.isinf(actual_p) else np.linalg.norm(edge_norms, ord=actual_p))

    return {
        "energy": energy,
        "p": float(np.inf if psymbol == -1 else psymbol),
        "n": float(n),
        "num_vars": float(xmosek.size),
    }


if __name__ == "__main__":
    out = mosek_soft_cross_fields_test(seed=0)
    print(out)
