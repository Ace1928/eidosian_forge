from sympy.utilities.iterables import sift
from .util import new
def sort_rl(expr):
    return new(expr.__class__, *sorted(expr.args, key=key))