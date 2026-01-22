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
def test_add_1():
    assert Order(x + x) == Order(x)
    assert Order(3 * x - 2 * x ** 2) == Order(x)
    assert Order(1 + x) == Order(1, x)
    assert Order(1 + 1 / x) == Order(1 / x)
    assert Order(log(x) + 1 / log(x)) == Order((log(x) ** 2 + 1) / log(x))
    assert Order(exp(1 / x) + x) == Order(exp(1 / x))
    assert Order(exp(1 / x) + 1 / x ** 20) == Order(exp(1 / x))