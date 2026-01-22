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
def test_order_at_infinity():
    assert Order(1 + x, (x, oo)) == Order(x, (x, oo))
    assert Order(3 * x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) * 3 == Order(x, (x, oo))
    assert -28 * Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(Order(x, (x, oo)), (x, oo)) == Order(x, (x, oo))
    assert Order(Order(x, (x, oo)), (y, oo)) == Order(x, (x, oo), (y, oo))
    assert Order(3, (x, oo)) == Order(1, (x, oo))
    assert Order(x ** 2 + x + y, (x, oo)) == O(x ** 2, (x, oo))
    assert Order(x ** 2 + x + y, (y, oo)) == O(y, (y, oo))
    assert Order(2 * x, (x, oo)) * x == Order(x ** 2, (x, oo))
    assert Order(2 * x, (x, oo)) / x == Order(1, (x, oo))
    assert Order(2 * x, (x, oo)) * x * exp(1 / x) == Order(x ** 2 * exp(1 / x), (x, oo))
    assert Order(2 * x, (x, oo)) * x * exp(1 / x) / log(x) ** 3 == Order(x ** 2 * exp(1 / x) * log(x) ** (-3), (x, oo))
    assert Order(x, (x, oo)) + 1 / x == 1 / x + Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + 1 == 1 + Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + x == x + Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + x ** 2 == x ** 2 + Order(x, (x, oo))
    assert Order(1 / x, (x, oo)) + 1 / x ** 2 == 1 / x ** 2 + Order(1 / x, (x, oo)) == Order(1 / x, (x, oo))
    assert Order(x, (x, oo)) + exp(1 / x) == exp(1 / x) + Order(x, (x, oo))
    assert Order(x, (x, oo)) ** 2 == Order(x ** 2, (x, oo))
    assert Order(x, (x, oo)) + Order(x ** 2, (x, oo)) == Order(x ** 2, (x, oo))
    assert Order(x, (x, oo)) + Order(x ** (-2), (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + Order(1 / x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) - Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + Order(1, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + Order(x ** 2, (x, oo)) == Order(x ** 2, (x, oo))
    assert Order(1 / x, (x, oo)) + Order(1, (x, oo)) == Order(1, (x, oo))
    assert Order(x, (x, oo)) + Order(exp(1 / x), (x, oo)) == Order(x, (x, oo))
    assert Order(x ** 3, (x, oo)) + Order(exp(2 / x), (x, oo)) == Order(x ** 3, (x, oo))
    assert Order(x ** (-3), (x, oo)) + Order(exp(2 / x), (x, oo)) == Order(exp(2 / x), (x, oo))
    assert Order(exp(x), (x, oo)).expr == Order(2 * exp(x), (x, oo)).expr == exp(x)
    assert Order(y ** x, (x, oo)).expr == Order(2 * y ** x, (x, oo)).expr == exp(x * log(y))
    assert Order(1 / x - 3 / (3 * x + 2), (x, oo)).expr == x ** (-2)