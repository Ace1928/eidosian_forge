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
@cacheit
def limitinf(e, x):
    """Limit e(x) for x-> oo."""
    old = e
    if not e.has(x):
        return e
    from sympy.simplify.powsimp import powdenest
    from sympy.calculus.util import AccumBounds
    if e.has(Order):
        e = e.expand().removeO()
    if not x.is_positive or x.is_integer:
        p = Dummy('p', positive=True)
        e = e.subs(x, p)
        x = p
    e = e.rewrite('tractable', deep=True, limitvar=x)
    e = powdenest(e)
    if isinstance(e, AccumBounds):
        if mrv_leadterm(e.min, x) != mrv_leadterm(e.max, x):
            raise NotImplementedError
        c0, e0 = mrv_leadterm(e.min, x)
    else:
        c0, e0 = mrv_leadterm(e, x)
    sig = sign(e0, x)
    if sig == 1:
        return S.Zero
    elif sig == -1:
        if c0.match(I * Wild('a', exclude=[I])):
            return c0 * oo
        s = sign(c0, x)
        if s == 0:
            raise ValueError('Leading term should not be 0')
        return s * oo
    elif sig == 0:
        if c0 == old:
            c0 = c0.cancel()
        return limitinf(c0, x)
    else:
        raise ValueError('{} could not be evaluated'.format(sig))