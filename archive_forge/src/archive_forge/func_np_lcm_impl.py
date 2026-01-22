import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_lcm_impl(context, builder, sig, args):
    xty, yty = sig.args
    assert xty == yty == sig.return_type
    x, y = args

    def lcm(a, b):
        """
        Like gcd, heavily cribbed from Julia.
        """
        return 0 if a == 0 else abs(a * (b // np.gcd(b, a)))
    res = context.compile_internal(builder, lcm, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)