from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun
def eulernum(ctx, n, exact=False):
    n = int(n)
    if exact:
        return int(ctx._eulernum(n))
    if n < 100:
        return ctx.mpf(ctx._eulernum(n))
    if n % 2:
        return ctx.zero
    return ctx.ldexp(ctx.eulerpoly(n, 0.5), n)