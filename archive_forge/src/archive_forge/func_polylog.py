from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun_wrapped
def polylog(ctx, s, z):
    s = ctx.convert(s)
    z = ctx.convert(z)
    if z == 1:
        return ctx.zeta(s)
    if z == -1:
        return -ctx.altzeta(s)
    if s == 0:
        return z / (1 - z)
    if s == 1:
        return -ctx.ln(1 - z)
    if s == -1:
        return z / (1 - z) ** 2
    if abs(z) <= 0.75 or (not ctx.isint(s) and abs(z) < 0.9):
        return polylog_series(ctx, s, z)
    if abs(z) >= 1.4 and ctx.isint(s):
        return (-1) ** (s + 1) * polylog_series(ctx, s, 1 / z) + polylog_continuation(ctx, int(ctx.re(s)), z)
    if ctx.isint(s):
        return polylog_unitcircle(ctx, int(ctx.re(s)), z)
    return polylog_general(ctx, s, z)