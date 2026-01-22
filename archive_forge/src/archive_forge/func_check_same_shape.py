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
def check_same_shape(*args, allow_cpu_scalar_tensors: bool):
    """
    Checks that all Tensors in args have the same shape.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensor objects in args have different devices
    """
    shape = None
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                continue
            if shape is None:
                shape = arg.shape
            if not is_same_shape(shape, arg.shape):
                msg = f'Shape {arg.shape} is not the expected shape {shape}!'
                raise RuntimeError(msg)
        else:
            msg = 'Unexpected type when checking for same shape, ' + str(type(arg)) + '!'
            raise RuntimeError(msg)