import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_complex_rint_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    in1 = context.make_complex(builder, ty, value=args[0])
    out = context.make_complex(builder, ty)
    inner_sig = typing.signature(*[float_ty] * 2)
    out.real = np_real_rint_impl(context, builder, inner_sig, [in1.real])
    out.imag = np_real_rint_impl(context, builder, inner_sig, [in1.imag])
    return out._getvalue()