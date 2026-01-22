import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_real_mod_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)
    in1, in2 = args
    ty = sig.args[0]
    ZERO = context.get_constant(ty, 0.0)
    res = builder.frem(in1, in2)
    res_ne_zero = builder.fcmp_ordered('!=', res, ZERO)
    den_lt_zero = builder.fcmp_ordered('<', in2, ZERO)
    res_lt_zero = builder.fcmp_ordered('<', res, ZERO)
    needs_fixing = builder.and_(res_ne_zero, builder.xor(den_lt_zero, res_lt_zero))
    fix_value = builder.select(needs_fixing, in2, ZERO)
    return builder.fadd(res, fix_value)