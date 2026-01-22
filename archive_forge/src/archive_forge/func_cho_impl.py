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
def cho_impl(a):
    n = a.shape[-1]
    if a.shape[-2] != n:
        msg = 'Last 2 dimensions of the array must be square.'
        raise np.linalg.LinAlgError(msg)
    out = a.copy()
    if n == 0:
        return out
    r = numba_xxpotrf(kind, UP, n, out.ctypes, n)
    if r != 0:
        if r < 0:
            fatal_error_func()
            assert 0
        if r > 0:
            raise np.linalg.LinAlgError('Matrix is not positive definite.')
    for col in range(n):
        out[:col, col] = 0
    return out