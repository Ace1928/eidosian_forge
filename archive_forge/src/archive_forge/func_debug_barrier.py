from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def debug_barrier(_builder=None):
    """
    Insert a barrier to synchronize all threads in a block.
    """
    return semantic.debug_barrier(_builder)