from functools import reduce
from itertools import product
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import fuzzy_not, fuzzy_or, fuzzy_and
from sympy.core.mod import Mod
from sympy.core.numbers import oo, igcd, Rational
from sympy.core.relational import Eq, is_eq
from sympy.core.kind import NumberKind
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols, Symbol
from sympy.core.sympify import _sympify, sympify, _sympy_converter
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.logic.boolalg import And, Or
from .sets import Set, Interval, Union, FiniteSet, ProductSet, SetKind
from sympy.utilities.misc import filldedent
def normalize_theta_set(theta):
    """
    Normalize a Real Set `theta` in the interval `[0, 2\\pi)`. It returns
    a normalized value of theta in the Set. For Interval, a maximum of
    one cycle $[0, 2\\pi]$, is returned i.e. for theta equal to $[0, 10\\pi]$,
    returned normalized value would be $[0, 2\\pi)$. As of now intervals
    with end points as non-multiples of ``pi`` is not supported.

    Raises
    ======

    NotImplementedError
        The algorithms for Normalizing theta Set are not yet
        implemented.
    ValueError
        The input is not valid, i.e. the input is not a real set.
    RuntimeError
        It is a bug, please report to the github issue tracker.

    Examples
    ========

    >>> from sympy.sets.fancysets import normalize_theta_set
    >>> from sympy import Interval, FiniteSet, pi
    >>> normalize_theta_set(Interval(9*pi/2, 5*pi))
    Interval(pi/2, pi)
    >>> normalize_theta_set(Interval(-3*pi/2, pi/2))
    Interval.Ropen(0, 2*pi)
    >>> normalize_theta_set(Interval(-pi/2, pi/2))
    Union(Interval(0, pi/2), Interval.Ropen(3*pi/2, 2*pi))
    >>> normalize_theta_set(Interval(-4*pi, 3*pi))
    Interval.Ropen(0, 2*pi)
    >>> normalize_theta_set(Interval(-3*pi/2, -pi/2))
    Interval(pi/2, 3*pi/2)
    >>> normalize_theta_set(FiniteSet(0, pi, 3*pi))
    {0, pi}

    """
    from sympy.functions.elementary.trigonometric import _pi_coeff
    if theta.is_Interval:
        interval_len = theta.measure
        if interval_len >= 2 * S.Pi:
            if interval_len == 2 * S.Pi and theta.left_open and theta.right_open:
                k = _pi_coeff(theta.start)
                return Union(Interval(0, k * S.Pi, False, True), Interval(k * S.Pi, 2 * S.Pi, True, True))
            return Interval(0, 2 * S.Pi, False, True)
        k_start, k_end = (_pi_coeff(theta.start), _pi_coeff(theta.end))
        if k_start is None or k_end is None:
            raise NotImplementedError('Normalizing theta without pi as coefficient is not yet implemented')
        new_start = k_start * S.Pi
        new_end = k_end * S.Pi
        if new_start > new_end:
            return Union(Interval(S.Zero, new_end, False, theta.right_open), Interval(new_start, 2 * S.Pi, theta.left_open, True))
        else:
            return Interval(new_start, new_end, theta.left_open, theta.right_open)
    elif theta.is_FiniteSet:
        new_theta = []
        for element in theta:
            k = _pi_coeff(element)
            if k is None:
                raise NotImplementedError('Normalizing theta without pi as coefficient, is not Implemented.')
            else:
                new_theta.append(k * S.Pi)
        return FiniteSet(*new_theta)
    elif theta.is_Union:
        return Union(*[normalize_theta_set(interval) for interval in theta.args])
    elif theta.is_subset(S.Reals):
        raise NotImplementedError('Normalizing theta when, it is of type %s is not implemented' % type(theta))
    else:
        raise ValueError(' %s is not a real set' % theta)