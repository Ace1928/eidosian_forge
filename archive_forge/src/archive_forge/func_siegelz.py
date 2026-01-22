from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun_wrapped
def siegelz(ctx, t, **kwargs):
    d = int(kwargs.get('derivative', 0))
    t = ctx.convert(t)
    t1 = ctx._re(t)
    t2 = ctx._im(t)
    prec = ctx.prec
    try:
        if abs(t1) > 500 * prec and t2 ** 2 < t1:
            v = ctx.rs_z(t, d)
            if ctx._is_real_type(t):
                return ctx._re(v)
            return v
    except NotImplementedError:
        pass
    ctx.prec += 21
    e1 = ctx.expj(ctx.siegeltheta(t))
    z = ctx.zeta(0.5 + ctx.j * t)
    if d == 0:
        v = e1 * z
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    z1 = ctx.zeta(0.5 + ctx.j * t, derivative=1)
    theta1 = ctx.siegeltheta(t, derivative=1)
    if d == 1:
        v = ctx.j * e1 * (z1 + z * theta1)
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    z2 = ctx.zeta(0.5 + ctx.j * t, derivative=2)
    theta2 = ctx.siegeltheta(t, derivative=2)
    comb1 = theta1 ** 2 - ctx.j * theta2
    if d == 2:

        def terms():
            return [2 * z1 * theta1, z2, z * comb1]
        v = ctx.sum_accurately(terms, 1)
        v = -e1 * v
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    ctx.prec += 10
    z3 = ctx.zeta(0.5 + ctx.j * t, derivative=3)
    theta3 = ctx.siegeltheta(t, derivative=3)
    comb2 = theta1 ** 3 - 3 * ctx.j * theta1 * theta2 - theta3
    if d == 3:

        def terms():
            return [3 * theta1 * z2, 3 * z1 * comb1, z3 + z * comb2]
        v = ctx.sum_accurately(terms, 1)
        v = -ctx.j * e1 * v
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    z4 = ctx.zeta(0.5 + ctx.j * t, derivative=4)
    theta4 = ctx.siegeltheta(t, derivative=4)

    def terms():
        return [theta1 ** 4, -6 * ctx.j * theta1 ** 2 * theta2, -3 * theta2 ** 2, -4 * theta1 * theta3, ctx.j * theta4]
    comb3 = ctx.sum_accurately(terms, 1)
    if d == 4:

        def terms():
            return [6 * theta1 ** 2 * z2, -6 * ctx.j * z2 * theta2, 4 * theta1 * z3, 4 * z1 * comb2, z4, z * comb3]
        v = ctx.sum_accurately(terms, 1)
        v = e1 * v
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    if d > 4:
        h = lambda x: ctx.siegelz(x, derivative=4)
        return ctx.diff(h, t, n=d - 4)