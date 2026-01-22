from __future__ import annotations
import numpy as np
def is_positive_semidefinite_matrix(mat, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """Test if a matrix is positive semidefinite"""
    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    if not is_hermitian_matrix(mat, rtol=rtol, atol=atol):
        return False
    vals = np.linalg.eigvalsh(mat)
    for v in vals:
        if v < -atol:
            return False
    return True