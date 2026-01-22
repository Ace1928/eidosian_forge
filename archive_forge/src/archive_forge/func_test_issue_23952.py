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
def test_issue_23952():
    assert (x ** (y + z)).expand(force=True) == x ** y * x ** z
    one = Symbol('1', integer=True, prime=True, odd=True, positive=True)
    two = Symbol('2', integer=True, prime=True, even=True)
    e = two - one
    for b in (0, x):
        assert unchanged(Pow, b, e)
        assert unchanged(Pow, b, -e)
        assert unchanged(Pow, b, y - x)
        assert unchanged(Pow, b, 3 - x)
        assert (b ** e).expand().is_Pow
        assert (b ** (-e)).expand().is_Pow
        assert (b ** (y - x)).expand().is_Pow
        assert (b ** (3 - x)).expand().is_Pow
    nn1 = Symbol('nn1', nonnegative=True)
    nn2 = Symbol('nn2', nonnegative=True)
    nn3 = Symbol('nn3', nonnegative=True)
    assert (x ** (nn1 + nn2)).expand() == x ** nn1 * x ** nn2
    assert (x ** (-nn1 - nn2)).expand() == x ** (-nn1) * x ** (-nn2)
    assert unchanged(Pow, x, nn1 + nn2 - nn3)
    assert unchanged(Pow, x, 1 + nn2 - nn3)
    assert unchanged(Pow, x, nn1 - nn2)
    assert unchanged(Pow, x, 1 - nn2)
    assert unchanged(Pow, x, -1 + nn2)