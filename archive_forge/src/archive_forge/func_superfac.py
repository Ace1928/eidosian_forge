from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def superfac(ctx, z):
    return ctx.barnesg(z + 2)