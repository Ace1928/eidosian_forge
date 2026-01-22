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
def oneD_impl(x, ord=None):
    n = len(x)
    if n == 0:
        return 0.0
    if ord == 2:
        return _oneD_norm_2(x)
    elif ord == np.inf:
        ret = abs(x[0])
        for k in range(1, n):
            val = abs(x[k])
            if val > ret:
                ret = val
        return ret
    elif ord == -np.inf:
        ret = abs(x[0])
        for k in range(1, n):
            val = abs(x[k])
            if val < ret:
                ret = val
        return ret
    elif ord == 0:
        ret = 0.0
        for k in range(n):
            if x[k] != 0.0:
                ret += 1.0
        return ret
    elif ord == 1:
        ret = 0.0
        for k in range(n):
            ret += abs(x[k])
        return ret
    else:
        ret = 0.0
        for k in range(n):
            ret += abs(x[k]) ** ord
        return ret ** (1.0 / ord)