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
def test_rewrite_MaxMin_as_Piecewise():
    from sympy.core.symbol import symbols
    from sympy.functions.elementary.piecewise import Piecewise
    x, y, z, a, b = symbols('x y z a b', real=True)
    vx, vy, va = symbols('vx vy va')
    assert Max(a, b).rewrite(Piecewise) == Piecewise((a, a >= b), (b, True))
    assert Max(x, y, z).rewrite(Piecewise) == Piecewise((x, (x >= y) & (x >= z)), (y, y >= z), (z, True))
    assert Max(x, y, a, b).rewrite(Piecewise) == Piecewise((a, (a >= b) & (a >= x) & (a >= y)), (b, (b >= x) & (b >= y)), (x, x >= y), (y, True))
    assert Min(a, b).rewrite(Piecewise) == Piecewise((a, a <= b), (b, True))
    assert Min(x, y, z).rewrite(Piecewise) == Piecewise((x, (x <= y) & (x <= z)), (y, y <= z), (z, True))
    assert Min(x, y, a, b).rewrite(Piecewise) == Piecewise((a, (a <= b) & (a <= x) & (a <= y)), (b, (b <= x) & (b <= y)), (x, x <= y), (y, True))
    assert Max(vx, vy).rewrite(Piecewise) == Piecewise((vx, vx >= vy), (vy, True))
    assert Min(va, vx, vy).rewrite(Piecewise) == Piecewise((va, (va <= vx) & (va <= vy)), (vx, vx <= vy), (vy, True))