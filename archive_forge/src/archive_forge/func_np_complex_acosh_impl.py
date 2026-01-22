import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_complex_acosh_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    csig2 = typing.signature(*[ty] * 3)
    ONE = context.get_constant_generic(builder, ty, 1.0 + 0j)
    x = args[0]
    x_plus_one = numbers.complex_add_impl(context, builder, csig2, [x, ONE])
    x_minus_one = numbers.complex_sub_impl(context, builder, csig2, [x, ONE])
    sqrt_x_plus_one = np_complex_sqrt_impl(context, builder, sig, [x_plus_one])
    sqrt_x_minus_one = np_complex_sqrt_impl(context, builder, sig, [x_minus_one])
    prod_sqrt = numbers.complex_mul_impl(context, builder, csig2, [sqrt_x_plus_one, sqrt_x_minus_one])
    log_arg = numbers.complex_add_impl(context, builder, csig2, [x, prod_sqrt])
    return np_complex_log_impl(context, builder, sig, [log_arg])