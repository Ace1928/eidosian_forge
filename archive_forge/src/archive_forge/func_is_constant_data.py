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
def is_constant_data(variable_ids: set[int]) -> bool:
    """
        Does the TensorView only contain constant data?
        """
    return variable_ids == {Constant.ID.value}