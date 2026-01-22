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
@classmethod
def empty_with_shape(cls, shape: tuple[int, int]) -> TensorRepresentation:
    return cls(np.array([], dtype=float), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), shape)