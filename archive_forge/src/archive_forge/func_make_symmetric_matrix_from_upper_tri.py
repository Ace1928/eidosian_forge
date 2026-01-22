from __future__ import annotations
import numpy as np
def make_symmetric_matrix_from_upper_tri(val):
    """Given a symmetric matrix in upper triangular matrix form as flat array indexes as:
    [A_xx,A_yy,A_zz,A_xy,A_xz,A_yz]
    This will generate the full matrix:
    [[A_xx,A_xy,A_xz],[A_xy,A_yy,A_yz],[A_xz,A_yz,A_zz].
    """
    idx = [0, 3, 4, 1, 5, 2]
    val = np.array(val)[idx]
    mask = ~np.tri(3, k=-1, dtype=bool)
    out = np.zeros((3, 3), dtype=val.dtype)
    out[mask] = val
    out.T[mask] = val
    return out