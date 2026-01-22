import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_complex_sinh_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    fty = ty.underlying_float
    fsig1 = typing.signature(*[fty] * 2)
    x = context.make_complex(builder, ty, args[0])
    out = context.make_complex(builder, ty)
    xr = x.real
    xi = x.imag
    sxi = np_real_sin_impl(context, builder, fsig1, [xi])
    shxr = np_real_sinh_impl(context, builder, fsig1, [xr])
    cxi = np_real_cos_impl(context, builder, fsig1, [xi])
    chxr = np_real_cosh_impl(context, builder, fsig1, [xr])
    out.real = builder.fmul(cxi, shxr)
    out.imag = builder.fmul(sxi, chxr)
    return out._getvalue()