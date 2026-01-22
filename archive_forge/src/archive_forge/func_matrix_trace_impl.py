import contextlib
import warnings
from llvmlite import ir
import numpy as np
import operator
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError, \
from .arrayobj import make_array, _empty_nd_impl, array_copy
from numba.np import numpy_support as np_support
def matrix_trace_impl(a, offset=0):
    rows, cols = a.shape
    k = offset
    if k < 0:
        rows = rows + k
    if k > 0:
        cols = cols - k
    n = max(min(rows, cols), 0)
    ret = 0
    if k >= 0:
        for i in range(n):
            ret += a[i, k + i]
    else:
        for i in range(n):
            ret += a[i - k, i]
    return ret