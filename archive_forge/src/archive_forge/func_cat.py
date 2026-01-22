from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def cat(input, other, can_reorder=False, _builder=None):
    """
    Concatenate the given blocks

    :param input: The first input tensor.
    :type input:
    :param other: The second input tensor.
    :type other:
    :param reorder: Compiler hint. If true, the compiler is
        allowed to reorder elements while concatenating inputs.  Only use if the
        order does not matter (e.g., result is only used in reduction ops)
    """
    return semantic.cat(input, other, can_reorder, _builder)