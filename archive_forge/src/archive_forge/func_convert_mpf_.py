import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
def convert_mpf_(x, prec, rounding):
    if hasattr(x, '_mpf_'):
        return x._mpf_
    if isinstance(x, int_types):
        return from_int(x, prec, rounding)
    if isinstance(x, float):
        return from_float(x, prec, rounding)
    if isinstance(x, basestring):
        return from_str(x, prec, rounding)
    raise NotImplementedError