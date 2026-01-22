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
def get_stack_func(total_rows: int, offset: int) -> Callable:
    """
        Returns a function that takes in a tensor, modifies the shape of the tensor by extending
        it to total_rows, and then shifts the entries by offset along axis 0.
        """

    def stack_func(tensor, p):
        coo_repr = tensor.tocoo()
        m = coo_repr.shape[0] // p
        slices = coo_repr.row // m
        new_rows = coo_repr.row + (slices + 1) * offset
        new_rows = new_rows + slices * (total_rows - m - offset).astype(int)
        return sp.csc_matrix((coo_repr.data, (new_rows, coo_repr.col)), shape=(int(total_rows * p), tensor.shape[1]))
    return stack_func