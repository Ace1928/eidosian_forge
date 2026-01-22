from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun_wrapped
def primezeta(ctx, s):
    if ctx.isnan(s):
        return s
    if ctx.re(s) <= 0:
        raise ValueError('prime zeta function defined only for re(s) > 0')
    if s == 1:
        return ctx.inf
    if s == 0.5:
        return ctx.mpc(ctx.ninf, ctx.pi)
    r = ctx.re(s)
    if r > ctx.prec:
        return 0.5 ** s
    else:
        wp = ctx.prec + int(r)

        def terms():
            orig = ctx.prec
            k = 0
            while 1:
                k += 1
                u = ctx.moebius(k)
                if not u:
                    continue
                ctx.prec = wp
                t = u * ctx.ln(ctx.zeta(k * s)) / k
                if not t:
                    return
                ctx.prec = orig
                yield t
    return ctx.sum_accurately(terms)