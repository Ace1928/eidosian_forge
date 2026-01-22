from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational as R, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.order import O
from sympy.simplify.radsimp import expand_numer
from sympy.core.function import expand, expand_multinomial, expand_power_base
from sympy.testing.pytest import raises
from sympy.core.random import verify_numerically
from sympy.abc import x, y, z
def test_expand_negative_integer_powers():
    expr = (x + y) ** (-2)
    assert expr.expand() == 1 / (2 * x * y + x ** 2 + y ** 2)
    assert expr.expand(multinomial=False) == (x + y) ** (-2)
    expr = (x + y) ** (-3)
    assert expr.expand() == 1 / (3 * x * x * y + 3 * x * y * y + x ** 3 + y ** 3)
    assert expr.expand(multinomial=False) == (x + y) ** (-3)
    expr = (x + y) ** 2 * (x + y) ** (-4)
    assert expr.expand() == 1 / (2 * x * y + x ** 2 + y ** 2)
    assert expr.expand(multinomial=False) == (x + y) ** (-2)