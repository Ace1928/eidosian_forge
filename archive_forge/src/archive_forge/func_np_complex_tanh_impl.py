import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_complex_tanh_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    fty = ty.underlying_float
    fsig1 = typing.signature(*[fty] * 2)
    ONE = context.get_constant(fty, 1.0)
    x = context.make_complex(builder, ty, args[0])
    out = context.make_complex(builder, ty)
    xr = x.real
    xi = x.imag
    si = np_real_sin_impl(context, builder, fsig1, [xi])
    ci = np_real_cos_impl(context, builder, fsig1, [xi])
    shr = np_real_sinh_impl(context, builder, fsig1, [xr])
    chr_ = np_real_cosh_impl(context, builder, fsig1, [xr])
    rs = builder.fmul(ci, shr)
    is_ = builder.fmul(si, chr_)
    rc = builder.fmul(ci, chr_)
    ic = builder.fmul(si, shr)
    sqr_rc = builder.fmul(rc, rc)
    sqr_ic = builder.fmul(ic, ic)
    d = builder.fadd(sqr_rc, sqr_ic)
    inv_d = builder.fdiv(ONE, d)
    rs_rc = builder.fmul(rs, rc)
    is_ic = builder.fmul(is_, ic)
    is_rc = builder.fmul(is_, rc)
    rs_ic = builder.fmul(rs, ic)
    numr = builder.fadd(rs_rc, is_ic)
    numi = builder.fsub(is_rc, rs_ic)
    out.real = builder.fmul(numr, inv_d)
    out.imag = builder.fmul(numi, inv_d)
    return out._getvalue()