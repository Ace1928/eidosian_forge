import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def to_special(u: np.ndarray) -> np.ndarray:
    """Converts a unitary matrix to a special unitary matrix.

    All unitary matrices u have |det(u)| = 1.
    Also for all d dimensional unitary matrix u, and scalar s:
        det(u * s) = det(u) * s^(d)
    To find a special unitary matrix from u:
        u * det(u)^{-1/d}

    Args:
        u: the unitary matrix
    Returns:
        the special unitary matrix
    """
    return u * np.linalg.det(u) ** (-1 / len(u))