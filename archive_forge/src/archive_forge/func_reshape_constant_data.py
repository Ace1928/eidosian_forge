from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
import numpy as np
import scipy.sparse as sp
from scipy.signal import convolve
from cvxpy.lin_ops import LinOp
from cvxpy.settings import (
@staticmethod
def reshape_constant_data(constant_data: dict[int, sp.csc_matrix], lin_op_shape: tuple[int, int]) -> dict[int, sp.csc_matrix]:
    """
        Reshape constant data from column format to the required shape for operations that
        do not require column format. This function unpacks the constant data dict and reshapes
        the stacked slices of the tensor 'v' according to the lin_op_shape argument.
        """
    return {k: SciPyCanonBackend._reshape_single_constant_tensor(v, lin_op_shape) for k, v in constant_data.items()}