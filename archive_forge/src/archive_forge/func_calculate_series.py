from functools import reduce
from sympy.core import Basic, S, Mul, PoleError, expand_mul
from sympy.core.cache import cacheit
from sympy.core.numbers import ilcm, I, oo
from sympy.core.symbol import Dummy, Wild
from sympy.core.traversal import bottom_up
from sympy.functions import log, exp, sign as _sign
from sympy.series.order import Order
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.misc import debug_decorator as debug
from sympy.utilities.timeutils import timethis
@debug
@timeit
def calculate_series(e, x, logx=None):
    """ Calculates at least one term of the series of ``e`` in ``x``.

    This is a place that fails most often, so it is in its own function.
    """
    SymPyDeprecationWarning(feature='calculate_series', useinstead='series() with suitable n, or as_leading_term', issue=21838, deprecated_since_version='1.12').warn()
    from sympy.simplify.powsimp import powdenest
    for t in e.lseries(x, logx=logx):
        t = bottom_up(t, lambda w: getattr(w, 'normal', lambda: w)())
        t = t.factor()
        if t.has(exp) and t.has(log):
            t = powdenest(t)
        if not t.is_zero:
            break
    return t