import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def can_numpy_support_shape(shape: Sequence[int]) -> bool:
    """Returns whether numpy supports the given shape or not numpy/numpy#5744."""
    return len(shape) <= _NPY_MAXDIMS