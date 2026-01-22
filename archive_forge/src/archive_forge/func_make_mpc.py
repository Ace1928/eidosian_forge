from .libmp.backend import basestring, exec_
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import rational
from . import function_docs
def make_mpc(ctx, v):
    a = new(ctx.mpc)
    a._mpc_ = v
    return a