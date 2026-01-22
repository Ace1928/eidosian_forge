from sympy.core.containers import Tuple
from sympy.core.numbers import pi
from sympy.core.power import Pow
from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.printing.str import sstr
from sympy.physics.units import (
from sympy.physics.units.util import convert_to, check_dimensions
from sympy.testing.pytest import raises
def test_eval_simplify():
    from sympy.physics.units import cm, mm, km, m, K, kilo
    from sympy.core.symbol import symbols
    x, y = symbols('x y')
    assert (cm / mm).simplify() == 10
    assert (km / m).simplify() == 1000
    assert (km / cm).simplify() == 100000
    assert (10 * x * K * km ** 2 / m / cm).simplify() == 1000000000 * x * kelvin
    assert (cm / km / m).simplify() == 1 / (10000000 * centimeter)
    assert (3 * kilo * meter).simplify() == 3000 * meter
    assert (4 * kilo * meter / (2 * kilometer)).simplify() == 2
    assert (4 * kilometer ** 2 / (kilo * meter) ** 2).simplify() == 4