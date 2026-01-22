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
def test_simple_6():
    assert Order(x) - Order(x) == Order(x)
    assert Order(x) + Order(1) == Order(1)
    assert Order(x) + Order(x ** 2) == Order(x)
    assert Order(1 / x) + Order(1) == Order(1 / x)
    assert Order(x) + Order(exp(1 / x)) == Order(exp(1 / x))
    assert Order(x ** 3) + Order(exp(2 / x)) == Order(exp(2 / x))
    assert Order(x ** (-3)) + Order(exp(2 / x)) == Order(exp(2 / x))