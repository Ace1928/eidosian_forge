from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
@_add_math_1arg_docstr('square root')
def sqrt(x, _builder=None):
    x = _to_tensor(x, _builder)
    return semantic.sqrt(x, _builder)