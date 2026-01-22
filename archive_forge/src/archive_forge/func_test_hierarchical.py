from sympy.core.function import PoleError
from sympy.core.numbers import oo
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.order import O
from sympy.abc import x
from sympy.testing.pytest import raises
def test_hierarchical():
    e = sin(1 / x + exp(-x))
    assert e.aseries(x, n=3, hir=True) == -exp(-2 * x) * sin(1 / x) / 2 + exp(-x) * cos(1 / x) + sin(1 / x) + O(exp(-3 * x), (x, oo))
    e = sin(x) * cos(exp(-x))
    assert e.aseries(x, hir=True) == exp(-4 * x) * sin(x) / 24 - exp(-2 * x) * sin(x) / 2 + sin(x) + O(exp(-6 * x), (x, oo))
    raises(PoleError, lambda: e.aseries(x))