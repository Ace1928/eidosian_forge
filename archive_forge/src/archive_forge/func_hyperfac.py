from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun_wrapped
def hyperfac(ctx, z):
    if z == ctx.inf:
        return z
    if abs(z) > 5:
        extra = 4 * int(ctx.log(abs(z), 2))
    else:
        extra = 0
    ctx.prec += extra
    if not ctx._im(z) and ctx._re(z) < 0 and ctx.isint(ctx._re(z)):
        n = int(ctx.re(z))
        h = ctx.hyperfac(-n - 1)
        if (n + 1) // 2 & 1:
            h = -h
        if ctx._is_complex_type(z):
            return h + 0j
        return h
    zp1 = z + 1
    v = ctx.exp(z * ctx.loggamma(zp1))
    ctx.prec -= extra
    return v / ctx.barnesg(zp1)