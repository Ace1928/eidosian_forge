import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_real_ldexp_impl(context, builder, sig, args):
    x1, x2 = args
    ty1, ty2 = sig.args
    x2 = context.cast(builder, x2, ty2, types.intc)
    f_fi_sig = typing.signature(ty1, ty1, types.intc)
    return mathimpl.ldexp_impl(context, builder, f_fi_sig, (x1, x2))