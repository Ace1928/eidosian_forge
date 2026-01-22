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
def check_same_dtype(*args):
    """
    Checks that all Tensors in args have the same device and that all Numbers have the
    same corresponding Python type.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensors objects in args have different dtypes
      - two Number objects in args have different types
      - there are Tensors and Numbers in args, and one of those Tensors corresponding
          Python types is different from the type of one of those Numbers
    """
    full_dtype = None
    scalar_type = None
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if full_dtype is None:
                full_dtype = arg.dtype
            if scalar_type is None:
                scalar_type = dtype_to_type(arg.dtype)
            if full_dtype is not arg.dtype:
                msg = 'Tensor with dtype ' + str(arg.dtype) + ' is not the expected dtype of ' + str(full_dtype) + '!'
                raise RuntimeError(msg)
            arg_type = dtype_to_type(arg.dtype)
            if arg_type is not scalar_type:
                msg = 'Tensor with corresponding Python type ' + str(arg_type) + ' is not the expected type of ' + str(scalar_type) + '!'
                raise RuntimeError(msg)
        else:
            msg = 'Unexpected type when checking for same dtype, ' + str(type(arg)) + '!'
            raise RuntimeError(msg)