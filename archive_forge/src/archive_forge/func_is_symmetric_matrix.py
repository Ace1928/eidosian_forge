from __future__ import annotations
import numpy as np
def is_symmetric_matrix(op, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """Test if an array is a symmetric matrix"""
    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    mat = np.array(op)
    if mat.ndim != 2:
        return False
    return np.allclose(mat, mat.T, rtol=rtol, atol=atol)