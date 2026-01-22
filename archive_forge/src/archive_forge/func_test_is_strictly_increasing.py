from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.calculus.singularities import (
from sympy.sets import Interval, FiniteSet
from sympy.testing.pytest import raises
from sympy.abc import x, y
def test_is_strictly_increasing():
    """Test whether is_strictly_increasing returns correct value."""
    assert is_strictly_increasing(4 * x ** 3 - 6 * x ** 2 - 72 * x + 30, Interval.Ropen(-oo, -2))
    assert is_strictly_increasing(4 * x ** 3 - 6 * x ** 2 - 72 * x + 30, Interval.Lopen(3, oo))
    assert not is_strictly_increasing(4 * x ** 3 - 6 * x ** 2 - 72 * x + 30, Interval.open(-2, 3))
    assert not is_strictly_increasing(-x ** 2, Interval(0, oo))
    assert not is_strictly_decreasing(1)
    assert is_strictly_increasing(4 * x ** 3 - 6 * x ** 2 - 72 * x + 30, Interval.open(-2, 3)) is False