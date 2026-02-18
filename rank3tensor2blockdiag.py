from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def rank3tensor2blockdiag(As: np.ndarray) -> sp.csr_matrix:
    arr = np.asarray(As, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("As must have shape (n, n2, m).")
    m = arr.shape[2]
    blocks = [sp.csr_matrix(arr[:, :, k]) for k in range(m)]
    return sp.block_diag(blocks, format="csr")
