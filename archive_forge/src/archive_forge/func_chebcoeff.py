from ..libmp.backend import xrange
from .calculus import defun
def chebcoeff(ctx, f, a, b, j, N):
    s = ctx.mpf(0)
    h = ctx.mpf(0.5)
    for k in range(1, N + 1):
        t = ctx.cospi((k - h) / N)
        s += f(t * (b - a) * h + (b + a) * h) * ctx.cospi(j * (k - h) / N)
    return 2 * s / N