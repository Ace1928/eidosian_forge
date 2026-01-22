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
def test_racah():
    assert racah(3, 3, 3, 3, 3, 3) == Rational(-1, 14)
    assert racah(2, 2, 2, 2, 2, 2) == Rational(-3, 70)
    assert racah(7, 8, 7, 1, 7, 7, prec=4).is_Float
    assert racah(5.5, 7.5, 9.5, 6.5, 8, 9) == -719 * sqrt(598) / 1158924
    assert abs(racah(5.5, 7.5, 9.5, 6.5, 8, 9, prec=4) - -0.01517) < S('1e-4')