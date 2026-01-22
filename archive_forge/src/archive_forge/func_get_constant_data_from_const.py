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
def get_constant_data_from_const(lin_op: LinOp) -> sp.csr_matrix:
    """
        Extract the constant data from a LinOp node of type "*_const".
        """
    constant = sp.csr_matrix(lin_op.data)
    assert constant.shape == lin_op.shape
    return constant