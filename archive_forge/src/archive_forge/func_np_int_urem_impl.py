import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_int_urem_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)
    num, den = args
    ty = sig.args[0]
    ZERO = context.get_constant(ty, 0)
    den_not_zero = builder.icmp_unsigned('!=', ZERO, den)
    bb_no_if = builder.basic_block
    with cgutils.if_unlikely(builder, den_not_zero):
        bb_if = builder.basic_block
        mod = builder.urem(num, den)
    result = builder.phi(ZERO.type)
    result.add_incoming(ZERO, bb_no_if)
    result.add_incoming(mod, bb_if)
    return result