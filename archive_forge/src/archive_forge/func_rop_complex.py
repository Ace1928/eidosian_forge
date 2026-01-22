import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
def rop_complex(s, t):
    ctx = s.ctx
    if not isinstance(t, s.ctx._types):
        t = s.ctx.convert(t)
    return g_complex(ctx, t._mpci_, s._mpci_)