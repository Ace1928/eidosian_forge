from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun_wrapped
def barnesg(ctx, z):
    if ctx.isinf(z):
        if z == ctx.inf:
            return z
        return ctx.nan
    if ctx.isnan(z):
        return z
    if not ctx._im(z) and ctx._re(z) <= 0 and ctx.isint(ctx._re(z)):
        return z * 0
    if abs(z) > 5:
        ctx.dps += 2 * ctx.log(abs(z), 2)
    if ctx.re(z) < -ctx.dps:
        w = 1 - z
        pi2 = 2 * ctx.pi
        u = ctx.expjpi(2 * w)
        v = ctx.j * ctx.pi / 12 - ctx.j * ctx.pi * w ** 2 / 2 + w * ctx.ln(1 - u) - ctx.j * ctx.polylog(2, u) / pi2
        v = ctx.barnesg(2 - z) * ctx.exp(v) / pi2 ** w
        if ctx._is_real_type(z):
            v = ctx._re(v)
        return v
    N = ctx.dps // 2 + 5
    G = 1
    while abs(z) < N or ctx.re(z) < 1:
        G /= ctx.gamma(z)
        z += 1
    z -= 1
    s = ctx.mpf(1) / 12
    s -= ctx.log(ctx.glaisher)
    s += z * ctx.log(2 * ctx.pi) / 2
    s += (z ** 2 / 2 - ctx.mpf(1) / 12) * ctx.log(z)
    s -= 3 * z ** 2 / 4
    z2k = z2 = z ** 2
    for k in xrange(1, N + 1):
        t = ctx.bernoulli(2 * k + 2) / (4 * k * (k + 1) * z2k)
        if abs(t) < ctx.eps:
            break
        z2k *= z2
        s += t
    return G * ctx.exp(s)