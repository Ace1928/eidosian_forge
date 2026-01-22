from ..libmp.backend import xrange
import math
import cmath
@defun
def unitroots(ctx, n, primitive=False):
    gcd = ctx._gcd
    prec = ctx.prec
    try:
        ctx.prec += 10
        if primitive:
            v = [ctx._rootof1(k, n) for k in range(n) if gcd(k, n) == 1]
        else:
            v = [ctx._rootof1(k, n) for k in range(n)]
    finally:
        ctx.prec = prec
    return [+x for x in v]