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
def real_impl(b, n, nrhs):
    res = np.empty(nrhs, dtype=real_dtype)
    for k in range(nrhs):
        res[k] = np.sum(b[n:, k] ** 2)
    return res