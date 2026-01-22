from typing import cast, List, Optional, Sequence, Union, Tuple
import numpy as np
from cirq.linalg import tolerance, transformations
from cirq import value
def is_special_orthogonal(matrix: np.ndarray, *, rtol: float=1e-05, atol: float=1e-08) -> bool:
    """Determines if a matrix is approximately special orthogonal.

    A matrix is special orthogonal if it is square and real and its transpose
    is its inverse and its determinant is one.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is special orthogonal within the given tolerance.
    """
    return is_orthogonal(matrix, rtol=rtol, atol=atol) and (matrix.shape[0] == 0 or np.allclose(np.linalg.det(matrix), 1, rtol=rtol, atol=atol))