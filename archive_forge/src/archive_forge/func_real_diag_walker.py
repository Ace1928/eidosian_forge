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
@register_jitable
def real_diag_walker(n, a, sgn):
    acc = 0.0
    for k in range(n):
        v = a[k, k]
        if v < 0.0:
            sgn = -sgn
            v = -v
        acc = acc + np.log(v)
    return (sgn + 0.0, acc)