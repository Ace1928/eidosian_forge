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
def test_clebsch_gordan():
    h = S.One
    k = S.Half
    l = Rational(3, 2)
    i = Rational(-1, 2)
    n = Rational(7, 2)
    p = Rational(5, 2)
    assert clebsch_gordan(k, k, 1, k, k, 1) == 1
    assert clebsch_gordan(k, k, 1, k, k, 0) == 0
    assert clebsch_gordan(k, k, 1, i, i, -1) == 1
    assert clebsch_gordan(k, k, 1, k, i, 0) == sqrt(2) / 2
    assert clebsch_gordan(k, k, 0, k, i, 0) == sqrt(2) / 2
    assert clebsch_gordan(k, k, 1, i, k, 0) == sqrt(2) / 2
    assert clebsch_gordan(k, k, 0, i, k, 0) == -sqrt(2) / 2
    assert clebsch_gordan(h, k, l, 1, k, l) == 1
    assert clebsch_gordan(h, k, l, 1, i, k) == 1 / sqrt(3)
    assert clebsch_gordan(h, k, k, 1, i, k) == sqrt(2) / sqrt(3)
    assert clebsch_gordan(h, k, k, 0, k, k) == -1 / sqrt(3)
    assert clebsch_gordan(h, k, l, 0, k, k) == sqrt(2) / sqrt(3)
    assert clebsch_gordan(h, h, S(2), 1, 1, S(2)) == 1
    assert clebsch_gordan(h, h, S(2), 1, 0, 1) == 1 / sqrt(2)
    assert clebsch_gordan(h, h, S(2), 0, 1, 1) == 1 / sqrt(2)
    assert clebsch_gordan(h, h, 1, 1, 0, 1) == 1 / sqrt(2)
    assert clebsch_gordan(h, h, 1, 0, 1, 1) == -1 / sqrt(2)
    assert clebsch_gordan(l, l, S(3), l, l, S(3)) == 1
    assert clebsch_gordan(l, l, S(2), l, k, S(2)) == 1 / sqrt(2)
    assert clebsch_gordan(l, l, S(3), l, k, S(2)) == 1 / sqrt(2)
    assert clebsch_gordan(S(2), S(2), S(4), S(2), S(2), S(4)) == 1
    assert clebsch_gordan(S(2), S(2), S(3), S(2), 1, S(3)) == 1 / sqrt(2)
    assert clebsch_gordan(S(2), S(2), S(3), 1, 1, S(2)) == 0
    assert clebsch_gordan(p, h, n, p, 1, n) == 1
    assert clebsch_gordan(p, h, p, p, 0, p) == sqrt(5) / sqrt(7)
    assert clebsch_gordan(p, h, l, k, 1, l) == 1 / sqrt(15)