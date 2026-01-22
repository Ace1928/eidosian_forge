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
def test_expand_power_base():
    assert expand_power_base((x * y * z) ** 4) == x ** 4 * y ** 4 * z ** 4
    assert expand_power_base((x * y * z) ** x).is_Pow
    assert expand_power_base((x * y * z) ** x, force=True) == x ** x * y ** x * z ** x
    assert expand_power_base((x * (y * z) ** 2) ** 3) == x ** 3 * y ** 6 * z ** 6
    assert expand_power_base((sin((x * y) ** 2) * y) ** z).is_Pow
    assert expand_power_base((sin((x * y) ** 2) * y) ** z, force=True) == sin((x * y) ** 2) ** z * y ** z
    assert expand_power_base((sin((x * y) ** 2) * y) ** z, deep=True) == (sin(x ** 2 * y ** 2) * y) ** z
    assert expand_power_base(exp(x) ** 2) == exp(2 * x)
    assert expand_power_base((exp(x) * exp(y)) ** 2) == exp(2 * x) * exp(2 * y)
    assert expand_power_base((exp((x * y) ** z) * exp(y)) ** 2) == exp(2 * (x * y) ** z) * exp(2 * y)
    assert expand_power_base((exp((x * y) ** z) * exp(y)) ** 2, deep=True, force=True) == exp(2 * x ** z * y ** z) * exp(2 * y)
    assert expand_power_base((exp(x) * exp(y)) ** z).is_Pow
    assert expand_power_base((exp(x) * exp(y)) ** z, force=True) == exp(x) ** z * exp(y) ** z