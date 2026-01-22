from .libmp.backend import xrange
from .libmp import int_types, sqrt_fixed
def round_fixed(x, prec):
    return x + (1 << prec - 1) >> prec << prec