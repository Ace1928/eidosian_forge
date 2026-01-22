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
def test_real_root():
    assert real_root(-8, 3) == -2
    assert real_root(-16, 4) == root(-16, 4)
    r = root(-7, 4)
    assert real_root(r) == r
    r1 = root(-1, 3)
    r2 = r1 ** 2
    r3 = root(-1, 4)
    assert real_root(r1 + r2 + r3) == -1 + r2 + r3
    assert real_root(root(-2, 3)) == -root(2, 3)
    assert real_root(-8.0, 3) == -2.0
    x = Symbol('x')
    n = Symbol('n')
    g = real_root(x, n)
    assert g.subs({'x': -8, 'n': 3}) == -2
    assert g.subs({'x': 8, 'n': 3}) == 2
    assert g.subs({'x': I, 'n': 3}) == cbrt(I)
    assert g.subs({'x': -8, 'n': 2}) == sqrt(-8)
    assert g.subs({'x': I, 'n': 2}) == sqrt(I)