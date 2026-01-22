from .ctx_base import StandardBaseContext
import math
import cmath
from . import math2
from . import function_docs
from .libmp import mpf_bernoulli, to_float, int_types
from . import libmp
def nint_distance(ctx, z):
    if hasattr(z, 'imag'):
        n = round(z.real)
    else:
        n = round(z)
    if n == z:
        return (n, ctx.ninf)
    return (n, ctx.mag(abs(z - n)))