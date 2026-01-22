from ..libmp.backend import xrange
from .calculus import defun
def emfun(point, tol):
    workprec = ctx.prec
    ctx.prec = prec + 10
    v = ctx.sumem(g, [point, ctx.inf], tol, error=1)
    ctx.prec = workprec
    return v