from sympy.core.add import Add
from sympy.core.function import (Function, expand)
from sympy.core.numbers import (I, Rational, nan, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (conjugate, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.series.order import O, Order
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises
from sympy.abc import w, x, y, z
def test_simple_1():
    o = Rational(0)
    assert Order(2 * x) == Order(x)
    assert Order(x) * 3 == Order(x)
    assert -28 * Order(x) == Order(x)
    assert Order(Order(x)) == Order(x)
    assert Order(Order(x), y) == Order(Order(x), x, y)
    assert Order(-23) == Order(1)
    assert Order(exp(x)) == Order(1, x)
    assert Order(exp(1 / x)).expr == exp(1 / x)
    assert Order(x * exp(1 / x)).expr == x * exp(1 / x)
    assert Order(x ** (o / 3)).expr == x ** (o / 3)
    assert Order(x ** (o * Rational(5, 3))).expr == x ** (o * Rational(5, 3))
    assert Order(x ** 2 + x + y, x) == O(1, x)
    assert Order(x ** 2 + x + y, y) == O(1, y)
    raises(ValueError, lambda: Order(exp(x), x, x))
    raises(TypeError, lambda: Order(x, 2 - x))