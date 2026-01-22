import math
import numbers
import numpy as np
import operator
from llvmlite import ir
from llvmlite.ir import Constant
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, errors, cgutils, optional
from numba.core.extending import intrinsic, overload_method
from numba.cpython.unsafe.numbers import viewer
def real_divmod_func_body(context, builder, vx, wx):
    pmod = cgutils.alloca_once(builder, vx.type)
    pdiv = cgutils.alloca_once(builder, vx.type)
    pfloordiv = cgutils.alloca_once(builder, vx.type)
    mod = builder.frem(vx, wx)
    div = builder.fdiv(builder.fsub(vx, mod), wx)
    builder.store(mod, pmod)
    builder.store(div, pdiv)
    ZERO = vx.type(0.0)
    NZERO = vx.type(-0.0)
    ONE = vx.type(1.0)
    mod_istrue = builder.fcmp_unordered('!=', mod, ZERO)
    wx_ltz = builder.fcmp_ordered('<', wx, ZERO)
    mod_ltz = builder.fcmp_ordered('<', mod, ZERO)
    with builder.if_else(mod_istrue, likely=True) as (if_nonzero_mod, if_zero_mod):
        with if_nonzero_mod:
            wx_ltz_ne_mod_ltz = builder.icmp_unsigned('!=', wx_ltz, mod_ltz)
            with builder.if_then(wx_ltz_ne_mod_ltz):
                builder.store(builder.fsub(div, ONE), pdiv)
                builder.store(builder.fadd(mod, wx), pmod)
        with if_zero_mod:
            mod = builder.select(wx_ltz, NZERO, ZERO)
            builder.store(mod, pmod)
    del mod, div
    div = builder.load(pdiv)
    div_istrue = builder.fcmp_ordered('!=', div, ZERO)
    with builder.if_then(div_istrue):
        realtypemap = {'float': types.float32, 'double': types.float64}
        realtype = realtypemap[str(wx.type)]
        floorfn = context.get_function(math.floor, typing.signature(realtype, realtype))
        floordiv = floorfn(builder, [div])
        floordivdiff = builder.fsub(div, floordiv)
        floordivincr = builder.fadd(floordiv, ONE)
        HALF = Constant(wx.type, 0.5)
        pred = builder.fcmp_ordered('>', floordivdiff, HALF)
        floordiv = builder.select(pred, floordivincr, floordiv)
        builder.store(floordiv, pfloordiv)
    with cgutils.ifnot(builder, div_istrue):
        div = builder.fmul(div, div)
        builder.store(div, pdiv)
        floordiv = builder.fdiv(builder.fmul(div, vx), wx)
        builder.store(floordiv, pfloordiv)
    return (builder.load(pfloordiv), builder.load(pmod))