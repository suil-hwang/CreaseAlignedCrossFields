# Crease Aligned Cross Fields (Python)

- [Paper](https://dl.acm.org/doi/abs/10.1145/3374209)
- [Talk](https://www.youtube.com/watch?v=a6Cv0tZulv4)
- [Fast forward](https://www.youtube.com/watch?v=M28EMpBRtnA)
- [Extra results](https://drive.google.com/file/d/1heg0i8wXiyBT-Zx9XVPOsEuLMNF4WuGM/view?usp=sharing)

## Overview

This repository provides a Python implementation for computing crease-aligned cross fields on triangle meshes.

## Dependencies

- Python 3.10+
- [MOSEK](https://www.mosek.com) (Python API)
- `numpy`, `scipy`
- `matplotlib` (only for visualization in `example.py`)

`arff` is used as a pip package from:

- `https://github.com/suil-hwang/arff/tree/master`

## Install

```bash
python -m pip install --upgrade "git+https://github.com/suil-hwang/arff.git@master"
```

If you run the optimization path, make sure MOSEK license/setup is valid in your environment.

## Run Example

```bash
python example.py
```

This runs `SolveLpCrossField` on `Meshes/notch5.obj`, computes the cross field, and writes a `.mat` result under `Results/`.

## Python API

Main solver:

- `SolveLpCrossField` in `SolveLpCrossField.py`

MATLAB-style signature:

```python
dirs1, fname, data = SolveLpCrossField(
    X, T, mname, n, p, Visualize, isFixedTriangle=None, fixedTriangleFrames=None
)
```

Key inputs:

- `X`: vertex array `(n, 3)`
- `T`: triangle array `(m, 3)`
- `n`: normal-alignment softness (`0` gives hard alignment)
- `p`: objective norm (`1`, `2`, or `np.inf`)
- `Visualize`: show final field if `True`

## Notes

- `arff/ext/ray` is MATLAB/MEX-side legacy layout.
- Python runtime uses the pip-installed `arff` package modules (`arff/src/ray` lineage).
