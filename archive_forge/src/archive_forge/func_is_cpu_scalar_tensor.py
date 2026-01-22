from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def is_cpu_scalar_tensor(a: Any) -> bool:
    return isinstance(a, TensorLike) and a.ndim == 0 and (a.device.type == 'cpu')