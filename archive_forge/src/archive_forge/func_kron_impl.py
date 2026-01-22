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
def kron_impl(a, b):
    aa = fix_a(a)
    bb = fix_b(b)
    am = aa.shape[-2]
    an = aa.shape[-1]
    bm = bb.shape[-2]
    bn = bb.shape[-1]
    cm = am * bm
    cn = an * bn
    C = np.empty((cm, cn), dtype=dt)
    for i in range(am):
        rjmp = i * bm
        for k in range(bm):
            irjmp = rjmp + k
            slc = bb[k, :]
            for j in range(an):
                cjmp = j * bn
                C[irjmp, cjmp:cjmp + bn] = aa[i, j] * slc
    return ret_c(a, b, C)