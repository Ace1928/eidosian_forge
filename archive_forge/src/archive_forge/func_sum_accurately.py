from .ctx_base import StandardBaseContext
import math
import cmath
from . import math2
from . import function_docs
from .libmp import mpf_bernoulli, to_float, int_types
from . import libmp
def sum_accurately(ctx, terms, check_step=1):
    s = ctx.zero
    k = 0
    for term in terms():
        s += term
        if not k % check_step and term:
            if abs(term) <= 1e-18 * abs(s):
                break
        k += 1
    return s