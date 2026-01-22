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
def check_is_matrix(A: TensorLikeType, f_name: str, arg_name: str='A'):
    torch._check(len(A.shape) >= 2, lambda: f'{f_name}: The input tensor {arg_name} must have at least 2 dimensions.')