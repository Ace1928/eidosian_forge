import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def ternary_op(self, lhs, rhs, other, op):
    return TensorHandle(op(lhs.data, rhs.data, other.data), other.dtype)