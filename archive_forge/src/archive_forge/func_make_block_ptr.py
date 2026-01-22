from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def make_block_ptr(base: tensor, shape, strides, offsets, block_shape, order, _builder=None):
    """
    Returns a pointer to a block in a parent tensor

    :param base: The base pointer to the parent tensor
    :param shape: The shape of the parent tensor
    :param strides: The strides of the parent tensor
    :param offsets: The offsets to the block
    :param block_shape: The shape of the block
    :param order: The order of the original data format
    """
    return semantic.make_block_ptr(base, shape, strides, offsets, block_shape, order, _builder)