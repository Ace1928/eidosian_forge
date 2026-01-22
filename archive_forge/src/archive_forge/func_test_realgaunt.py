from sympy.core.numbers import (I, pi, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import Matrix
from sympy.physics.wigner import (clebsch_gordan, wigner_9j, wigner_6j, gaunt,
from sympy.testing.pytest import raises
def test_realgaunt():
    for l in range(3):
        for m in range(-l, l + 1):
            assert real_gaunt(0, l, l, 0, m, m) == 1 / (2 * sqrt(pi))
    assert real_gaunt(1, 1, 2, 0, 0, 0) == sqrt(5) / (5 * sqrt(pi))
    assert real_gaunt(1, 1, 2, 1, 1, 0) == -sqrt(5) / (10 * sqrt(pi))
    assert real_gaunt(2, 2, 2, 0, 0, 0) == sqrt(5) / (7 * sqrt(pi))
    assert real_gaunt(2, 2, 2, 0, 2, 2) == -sqrt(5) / (7 * sqrt(pi))
    assert real_gaunt(2, 2, 2, -2, -2, 0) == -sqrt(5) / (7 * sqrt(pi))
    assert real_gaunt(1, 1, 2, -1, 0, -1) == sqrt(15) / (10 * sqrt(pi))
    assert real_gaunt(1, 1, 2, 0, 1, 1) == sqrt(15) / (10 * sqrt(pi))
    assert real_gaunt(1, 1, 2, 1, 1, 2) == sqrt(15) / (10 * sqrt(pi))
    assert real_gaunt(1, 1, 2, -1, 1, -2) == -sqrt(15) / (10 * sqrt(pi))
    assert real_gaunt(1, 1, 2, -1, -1, 2) == -sqrt(15) / (10 * sqrt(pi))
    assert real_gaunt(2, 2, 2, 0, 1, 1) == sqrt(5) / (14 * sqrt(pi))
    assert real_gaunt(2, 2, 2, 1, 1, 2) == sqrt(15) / (14 * sqrt(pi))
    assert real_gaunt(2, 2, 2, -1, -1, 2) == -sqrt(15) / (14 * sqrt(pi))
    assert real_gaunt(-2, -2, -2, -2, -2, 0) is S.Zero
    assert real_gaunt(-2, 1, 0, 1, 1, 1) is S.Zero
    assert real_gaunt(-2, -1, -2, -1, -1, 0) is S.Zero
    assert real_gaunt(-2, -2, -2, -2, -2, -2) is S.Zero
    assert real_gaunt(-2, -1, -2, -1, -1, -1) is S.Zero
    x = symbols('x', integer=True)
    v = [0] * 6
    for i in range(len(v)):
        v[i] = x
        raises(ValueError, lambda: real_gaunt(*v))
        v[i] = 0