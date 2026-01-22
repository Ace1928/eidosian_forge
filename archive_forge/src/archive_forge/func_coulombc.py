from .functions import defun, defun_wrapped
@defun_wrapped
def coulombc(ctx, l, eta, _cache={}):
    if (l, eta) in _cache and _cache[l, eta][0] >= ctx.prec:
        return +_cache[l, eta][1]
    G3 = ctx.loggamma(2 * l + 2)
    G1 = ctx.loggamma(1 + l + ctx.j * eta)
    G2 = ctx.loggamma(1 + l - ctx.j * eta)
    v = 2 ** l * ctx.exp((-ctx.pi * eta + G1 + G2) / 2 - G3)
    if not (ctx.im(l) or ctx.im(eta)):
        v = ctx.re(v)
    _cache[l, eta] = (ctx.prec, v)
    return v