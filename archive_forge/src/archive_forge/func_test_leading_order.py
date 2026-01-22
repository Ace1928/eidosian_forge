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
def test_leading_order():
    assert (x + 1 + 1 / x ** 5).extract_leading_order(x) == ((1 / x ** 5, O(1 / x ** 5)),)
    assert (1 + 1 / x).extract_leading_order(x) == ((1 / x, O(1 / x)),)
    assert (1 + x).extract_leading_order(x) == ((1, O(1, x)),)
    assert (1 + x ** 2).extract_leading_order(x) == ((1, O(1, x)),)
    assert (2 + x ** 2).extract_leading_order(x) == ((2, O(1, x)),)
    assert (x + x ** 2).extract_leading_order(x) == ((x, O(x)),)