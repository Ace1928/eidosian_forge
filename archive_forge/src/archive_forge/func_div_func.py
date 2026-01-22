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
def div_func(x, p):
    if p == 1:
        return lhs.multiply(x)
    else:
        new_lhs = sp.vstack([lhs] * p)
        return new_lhs.multiply(x)