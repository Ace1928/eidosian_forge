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
def test_power_expand():
    """Test for Pow.expand()"""
    a = Symbol('a')
    b = Symbol('b')
    p = (a + b) ** 2
    assert p.expand() == a ** 2 + b ** 2 + 2 * a * b
    p = (1 + 2 * (1 + a)) ** 2
    assert p.expand() == 9 + 4 * a ** 2 + 12 * a
    p = 2 ** (a + b)
    assert p.expand() == 2 ** a * 2 ** b
    A = Symbol('A', commutative=False)
    B = Symbol('B', commutative=False)
    assert (2 ** (A + B)).expand() == 2 ** (A + B)
    assert (A ** (a + b)).expand() != A ** (a + b)