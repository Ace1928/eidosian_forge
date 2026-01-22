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
def test_issue_11099():
    from sympy.abc import x, y
    fixed_test_data = {x: -2, y: 3}
    assert Min(x, y).evalf(subs=fixed_test_data) == Min(x, y).subs(fixed_test_data).evalf()
    assert Max(x, y).evalf(subs=fixed_test_data) == Max(x, y).subs(fixed_test_data).evalf()
    from sympy.core.random import randint
    for i in range(20):
        random_test_data = {x: randint(-100, 100), y: randint(-100, 100)}
        assert Min(x, y).evalf(subs=random_test_data) == Min(x, y).subs(random_test_data).evalf()
        assert Max(x, y).evalf(subs=random_test_data) == Max(x, y).subs(random_test_data).evalf()