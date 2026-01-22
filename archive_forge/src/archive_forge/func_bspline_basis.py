from sympy.core import S, sympify
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions import Piecewise, piecewise_fold
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from functools import lru_cache
@lru_cache(maxsize=128)
def bspline_basis(d, knots, n, x):
    """
    The $n$-th B-spline at $x$ of degree $d$ with knots.

    Explanation
    ===========

    B-Splines are piecewise polynomials of degree $d$. They are defined on a
    set of knots, which is a sequence of integers or floats.

    Examples
    ========

    The 0th degree splines have a value of 1 on a single interval:

        >>> from sympy import bspline_basis
        >>> from sympy.abc import x
        >>> d = 0
        >>> knots = tuple(range(5))
        >>> bspline_basis(d, knots, 0, x)
        Piecewise((1, (x >= 0) & (x <= 1)), (0, True))

    For a given ``(d, knots)`` there are ``len(knots)-d-1`` B-splines
    defined, that are indexed by ``n`` (starting at 0).

    Here is an example of a cubic B-spline:

        >>> bspline_basis(3, tuple(range(5)), 0, x)
        Piecewise((x**3/6, (x >= 0) & (x <= 1)),
                  (-x**3/2 + 2*x**2 - 2*x + 2/3,
                  (x >= 1) & (x <= 2)),
                  (x**3/2 - 4*x**2 + 10*x - 22/3,
                  (x >= 2) & (x <= 3)),
                  (-x**3/6 + 2*x**2 - 8*x + 32/3,
                  (x >= 3) & (x <= 4)),
                  (0, True))

    By repeating knot points, you can introduce discontinuities in the
    B-splines and their derivatives:

        >>> d = 1
        >>> knots = (0, 0, 2, 3, 4)
        >>> bspline_basis(d, knots, 0, x)
        Piecewise((1 - x/2, (x >= 0) & (x <= 2)), (0, True))

    It is quite time consuming to construct and evaluate B-splines. If
    you need to evaluate a B-spline many times, it is best to lambdify them
    first:

        >>> from sympy import lambdify
        >>> d = 3
        >>> knots = tuple(range(10))
        >>> b0 = bspline_basis(d, knots, 0, x)
        >>> f = lambdify(x, b0)
        >>> y = f(0.5)

    Parameters
    ==========

    d : integer
        degree of bspline

    knots : list of integer values
        list of knots points of bspline

    n : integer
        $n$-th B-spline

    x : symbol

    See Also
    ========

    bspline_basis_set

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/B-spline

    """
    xvar = x
    x = Dummy()
    knots = tuple((sympify(k) for k in knots))
    d = int(d)
    n = int(n)
    n_knots = len(knots)
    n_intervals = n_knots - 1
    if n + d + 1 > n_intervals:
        raise ValueError('n + d + 1 must not exceed len(knots) - 1')
    if d == 0:
        result = Piecewise((S.One, Interval(knots[n], knots[n + 1]).contains(x)), (0, True))
    elif d > 0:
        denom = knots[n + d + 1] - knots[n + 1]
        if denom != S.Zero:
            B = (knots[n + d + 1] - x) / denom
            b2 = bspline_basis(d - 1, knots, n + 1, x)
        else:
            b2 = B = S.Zero
        denom = knots[n + d] - knots[n]
        if denom != S.Zero:
            A = (x - knots[n]) / denom
            b1 = bspline_basis(d - 1, knots, n, x)
        else:
            b1 = A = S.Zero
        result = _add_splines(A, b1, B, b2, x)
    else:
        raise ValueError('degree must be non-negative: %r' % n)
    return result.xreplace({x: xvar})