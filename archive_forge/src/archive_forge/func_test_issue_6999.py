from sympy.core.numbers import E
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.order import Order
from sympy.abc import x, y
def test_issue_6999():
    s = tanh(x).lseries(x, 1)
    assert next(s) == tanh(1)
    assert next(s) == x - (x - 1) * tanh(1) ** 2 - 1
    assert next(s) == -(x - 1) ** 2 * tanh(1) + (x - 1) ** 2 * tanh(1) ** 3
    assert next(s) == -(x - 1) ** 3 * tanh(1) ** 4 - (x - 1) ** 3 / 3 + 4 * (x - 1) ** 3 * tanh(1) ** 2 / 3