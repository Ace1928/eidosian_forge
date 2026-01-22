import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_complex_reciprocal_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    ZERO = context.get_constant(float_ty, 0.0)
    ONE = context.get_constant(float_ty, 1.0)
    in1 = context.make_complex(builder, ty, value=args[0])
    out = context.make_complex(builder, ty)
    in1r = in1.real
    in1i = in1.imag
    in1r_abs = _fabs(context, builder, in1r)
    in1i_abs = _fabs(context, builder, in1i)
    in1i_abs_le_in1r_abs = builder.fcmp_ordered('<=', in1i_abs, in1r_abs)
    with builder.if_else(in1i_abs_le_in1r_abs) as (then, otherwise):
        with then:
            r = builder.fdiv(in1i, in1r)
            tmp0 = builder.fmul(in1i, r)
            d = builder.fadd(in1r, tmp0)
            inv_d = builder.fdiv(ONE, d)
            minus_r = builder.fsub(ZERO, r)
            out.real = inv_d
            out.imag = builder.fmul(minus_r, inv_d)
        with otherwise:
            r = builder.fdiv(in1r, in1i)
            tmp0 = builder.fmul(in1r, r)
            d = builder.fadd(tmp0, in1i)
            inv_d = builder.fdiv(ONE, d)
            out.real = builder.fmul(r, inv_d)
            out.imag = builder.fsub(ZERO, inv_d)
    return out._getvalue()