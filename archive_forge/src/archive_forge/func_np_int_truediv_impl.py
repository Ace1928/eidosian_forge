import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_int_truediv_impl(context, builder, sig, args):
    num, den = args
    lltype = num.type
    assert all((i.type == lltype for i in args)), 'must have homogeneous types'
    numty, denty = sig.args
    num = context.cast(builder, num, numty, types.float64)
    den = context.cast(builder, den, denty, types.float64)
    return builder.fdiv(num, den)