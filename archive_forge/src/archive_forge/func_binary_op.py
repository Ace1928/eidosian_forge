import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def binary_op(self, lhs, rhs, op):
    return TensorHandle(op(lhs.data, rhs.data), lhs.dtype)