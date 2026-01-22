import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_complex_sign_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    op = args[0]
    ty = sig.args[0]
    float_ty = ty.underlying_float
    ZERO = context.get_constant(float_ty, 0.0)
    ONE = context.get_constant(float_ty, 1.0)
    MINUS_ONE = context.get_constant(float_ty, -1.0)
    NAN = context.get_constant(float_ty, float('nan'))
    result = context.make_complex(builder, ty)
    result.real = ZERO
    result.imag = ZERO
    cmp_sig = typing.signature(types.boolean, *[ty] * 2)
    cmp_args = [op, result._getvalue()]
    arg1_ge_arg2 = np_complex_ge_impl(context, builder, cmp_sig, cmp_args)
    arg1_eq_arg2 = np_complex_eq_impl(context, builder, cmp_sig, cmp_args)
    arg1_lt_arg2 = np_complex_lt_impl(context, builder, cmp_sig, cmp_args)
    real_when_ge = builder.select(arg1_eq_arg2, ZERO, ONE)
    real_when_nge = builder.select(arg1_lt_arg2, MINUS_ONE, NAN)
    result.real = builder.select(arg1_ge_arg2, real_when_ge, real_when_nge)
    return result._getvalue()