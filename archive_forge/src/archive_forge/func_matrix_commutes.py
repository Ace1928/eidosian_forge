from typing import cast, List, Optional, Sequence, Union, Tuple
import numpy as np
from cirq.linalg import tolerance, transformations
from cirq import value
def matrix_commutes(m1: np.ndarray, m2: np.ndarray, *, rtol: float=1e-05, atol: float=1e-08) -> bool:
    """Determines if two matrices approximately commute.

    Two matrices A and B commute if they are square and have the same size and
    AB = BA.

    Args:
        m1: One of the matrices.
        m2: The other matrix.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the two matrices have compatible sizes and a commutator equal
        to zero within tolerance.
    """
    return m1.shape[0] == m1.shape[1] and m1.shape == m2.shape and np.allclose(m1.dot(m2), m2.dot(m1), rtol=rtol, atol=atol)