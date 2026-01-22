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
def check_args(a, b, out):
    m, k = a.shape
    _k, n = b.shape
    if k != _k:
        raise ValueError('incompatible array sizes for np.dot(a, b) (matrix * matrix)')
    if out.shape != (m, n):
        raise ValueError('incompatible output array size for np.dot(a, b, out) (matrix * matrix)')