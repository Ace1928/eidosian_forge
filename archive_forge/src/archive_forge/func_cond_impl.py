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
@overload(np.linalg.cond)
def cond_impl(x, p=None):
    ensure_lapack()
    _check_linalg_matrix(x, 'cond')

    def impl(x, p=None):
        if p == 2 or p == -2 or p is None:
            s = _compute_singular_values(x)
            if p == 2 or p is None:
                r = np.divide(s[0], s[-1])
            else:
                r = np.divide(s[-1], s[0])
        else:
            norm_x = np.linalg.norm(x, p)
            norm_inv_x = np.linalg.norm(np.linalg.inv(x), p)
            r = norm_x * norm_inv_x
        if np.isnan(r):
            return np.inf
        else:
            return r
    return impl