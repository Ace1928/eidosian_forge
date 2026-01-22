import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
def create_Piecewise(parameter_name, nan_fallback=False):
    """
    Examples
    --------
    >>> Power = Expr.from_callback(lambda args, x, backend=None: args[0]*x**args[1],
    ...     argument_names=('scale', 'pow'), parameter_keys=('x',))
    >>> minus_x = Power([-1, 1])
    >>> cube = Power([1, 3])
    >>> PW = create_Piecewise('x')
    >>> pw = PW([-float('inf'), minus_x, 0, cube, float('inf')])
    >>> pw({'x': -5}) == 5
    True
    >>> pw({'x': 2}) == 8
    True

    """

    def _pw(bounds_exprs, x, backend=math, **kwargs):
        if len(bounds_exprs) < 3:
            raise ValueError('Need at least 3 args')
        if len(bounds_exprs) % 2 != 1:
            raise ValueError('Need an odd number of bounds/exprs')
        n_exprs = (len(bounds_exprs) - 1) // 2
        lower = [bounds_exprs[2 * (i + 0)] for i in range(n_exprs)]
        upper = [bounds_exprs[2 * (i + 1)] for i in range(n_exprs)]
        exprs = [bounds_exprs[2 * i + 1] for i in range(n_exprs)]
        try:
            pw = backend.Piecewise
        except AttributeError:
            for lo, up, ex in zip(lower, upper, exprs):
                if lo <= x <= up:
                    return ex
            else:
                raise ValueError('not within any bounds: %s' % x)
        else:
            _NAN = backend.Symbol('NAN')
            return pw(*[(ex, backend.And(lo <= x, x <= up)) for lo, up, ex in zip(lower, upper, exprs)] + ([(_NAN, True)] if nan_fallback else []))
    return Expr.from_callback(_pw, parameter_keys=(parameter_name,))