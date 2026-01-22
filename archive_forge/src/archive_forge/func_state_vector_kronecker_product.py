import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def state_vector_kronecker_product(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Merges two state vectors into a single unified state vector.

    The resulting vector's shape will be `t1.shape + t2.shape`.

    Args:
        t1: The first state vector.
        t2: The second state vector.
    Returns:
        A new state vector representing the unified state.
    """
    return np.outer(t1, t2).reshape(t1.shape + t2.shape)