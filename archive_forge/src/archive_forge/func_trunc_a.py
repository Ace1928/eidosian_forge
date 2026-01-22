import math
from .functions import defun
def trunc_a(t):
    wp = ctx.prec
    ctx.prec = wp + 2
    aa = ctx.sqrt(t / (2 * ctx.pi))
    ctx.prec = wp
    return aa