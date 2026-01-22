import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
@overload(fnoverload, target='generic')
def ol_npy_logaddexp(x1, x2):
    if x1 != x2:
        return
    shift = x1(const)

    def impl(x1, x2):
        x, y = (x1, x2)
        if x == y:
            return x + shift
        else:
            tmp = x - y
            if tmp > 0:
                return x + log1pfn(expfn(-tmp))
            elif tmp <= 0:
                return y + log1pfn(expfn(tmp))
            else:
                return tmp
    return impl