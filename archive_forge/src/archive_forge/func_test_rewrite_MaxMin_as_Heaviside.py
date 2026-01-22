import itertools as it
from sympy.core.expr import unchanged
from sympy.core.function import Function
from sympy.core.numbers import I, oo, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.external import import_module
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.miscellaneous import (sqrt, cbrt, root, Min,
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.delta_functions import Heaviside
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises, skip, ignore_warnings
def test_rewrite_MaxMin_as_Heaviside():
    from sympy.abc import x
    assert Max(0, x).rewrite(Heaviside) == x * Heaviside(x)
    assert Max(3, x).rewrite(Heaviside) == x * Heaviside(x - 3) + 3 * Heaviside(-x + 3)
    assert Max(0, x + 2, 2 * x).rewrite(Heaviside) == 2 * x * Heaviside(2 * x) * Heaviside(x - 2) + (x + 2) * Heaviside(-x + 2) * Heaviside(x + 2)
    assert Min(0, x).rewrite(Heaviside) == x * Heaviside(-x)
    assert Min(3, x).rewrite(Heaviside) == x * Heaviside(-x + 3) + 3 * Heaviside(x - 3)
    assert Min(x, -x, -2).rewrite(Heaviside) == x * Heaviside(-2 * x) * Heaviside(-x - 2) - x * Heaviside(2 * x) * Heaviside(x - 2) - 2 * Heaviside(-x + 2) * Heaviside(x + 2)