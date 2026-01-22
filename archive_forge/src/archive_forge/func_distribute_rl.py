from sympy.utilities.iterables import sift
from .util import new
def distribute_rl(expr):
    for i, arg in enumerate(expr.args):
        if isinstance(arg, B):
            first, b, tail = (expr.args[:i], expr.args[i], expr.args[i + 1:])
            return B(*[A(*first + (arg,) + tail) for arg in b.args])
    return expr