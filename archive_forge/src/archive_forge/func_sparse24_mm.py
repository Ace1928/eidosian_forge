import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
def sparse24_mm(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 2
    A, B = args
    if A.ndim != 2 or B.ndim != 2:
        raise NotImplementedError('`Sparse24Tensor` matmul: Broadcasting is not implemented')
    if isinstance(A, Sparse24Tensor):
        return A._mm(B)
    else:
        B_t = B.t()
        assert isinstance(B_t, Sparse24Tensor)
        return B_t._mm(A.t(), prefer_col_major_output=True).t()