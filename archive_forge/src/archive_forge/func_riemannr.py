from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun_wrapped
def riemannr(ctx, x):
    if x == 0:
        return ctx.zero
    if abs(x) > 1000:
        a = ctx.li(x)
        b = 0.5 * ctx.li(ctx.sqrt(x))
        if abs(b) < abs(a) * ctx.eps:
            return a
    if abs(x) < 0.01:
        ctx.prec += int(-ctx.log(abs(x), 2))
    s = t = ctx.one
    u = ctx.ln(x)
    k = 1
    while abs(t) > abs(s) * ctx.eps:
        t = t * u / k
        s += t / (k * ctx._zeta_int(k + 1))
        k += 1
    return s