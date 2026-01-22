import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_complex_log1p_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    float_unary_sig = typing.signature(*[float_ty] * 2)
    float_binary_sig = typing.signature(*[float_ty] * 3)
    ONE = context.get_constant(float_ty, 1.0)
    in1 = context.make_complex(builder, ty, value=args[0])
    out = context.make_complex(builder, ty)
    real_plus_one = builder.fadd(in1.real, ONE)
    l = np_real_hypot_impl(context, builder, float_binary_sig, [real_plus_one, in1.imag])
    out.imag = np_real_atan2_impl(context, builder, float_binary_sig, [in1.imag, real_plus_one])
    out.real = np_real_log_impl(context, builder, float_unary_sig, [l])
    return out._getvalue()