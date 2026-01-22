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
def get_param_slice(self, param_offset: int) -> sp.csc_matrix:
    """
        Returns a single slice of the tensor for a given parameter offset.
        """
    mask = self.parameter_offset == param_offset
    return sp.csc_matrix((self.data[mask], (self.row[mask], self.col[mask])), self.shape)