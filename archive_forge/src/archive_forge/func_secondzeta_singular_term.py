from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
def secondzeta_singular_term(ctx, s, a, **kwargs):
    factor = a ** (0.5 * (s - 1)) / (4 * ctx.sqrt(ctx.pi) * ctx.gamma(0.5 * s))
    extraprec = ctx.mag(factor)
    ctx.prec += extraprec
    factor = a ** (0.5 * (s - 1)) / (4 * ctx.sqrt(ctx.pi) * ctx.gamma(0.5 * s))
    tol = ctx.eps
    f = lambda n: ctx.bernpoly(n, 0.75) * (4 * ctx.sqrt(a)) ** n * ctx.gamma(0.5 * n) / ((s + n - 1) * ctx.fac(n))
    totsum = ctx.zero
    mg1 = ctx.inf
    n = 1
    term = f(n)
    mg2 = abs(term)
    while mg2 > tol and mg2 <= mg1:
        totsum += term
        n += 1
        term = f(n)
        totsum += term
        n += 1
        term = f(n)
        mg1 = mg2
        mg2 = abs(term)
    totsum += term
    pole = -2 * (s - 1) ** (-2) + (ctx.euler + ctx.log(16 * ctx.pi ** 2 * a)) * (s - 1) ** (-1)
    st = factor * (pole + totsum)
    err = 0
    if kwargs.get('error'):
        if not (mg2 > tol and mg2 <= mg1):
            if mg2 <= tol:
                err = ctx.mpf(10) ** int(ctx.log(abs(factor * tol), 10))
            if mg2 > mg1:
                err = ctx.mpf(10) ** int(ctx.log(abs(factor * mg1), 10))
        err = max(err, ctx.eps * 1.0)
    ctx.prec -= extraprec
    return (+st, err)