from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
def secondzeta_exp_term(ctx, s, a):
    if ctx.isint(s) and ctx.re(s) <= 0:
        m = int(round(ctx.re(s)))
        if not m & 1:
            return ctx.mpf('-0.25') ** (-m // 2)
    tol = ctx.eps
    f = lambda n: (0.25 * a) ** n / ((n + 0.5 * s) * ctx.fac(n))
    totsum = ctx.zero
    term = f(0)
    mg = ctx.inf
    n = 0
    while mg > tol:
        totsum += term
        n += 1
        term = f(n)
        mg = abs(term)
    v = a ** (0.5 * s) * totsum / ctx.gamma(0.5 * s)
    return v