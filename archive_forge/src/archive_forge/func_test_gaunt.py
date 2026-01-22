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
def test_gaunt():

    def tn(a, b):
        return (a - b).n(64) < S('1e-64')
    assert gaunt(1, 0, 1, 1, 0, -1) == -1 / (2 * sqrt(pi))
    assert isinstance(gaunt(1, 1, 0, -1, 1, 0).args[0], Rational)
    assert isinstance(gaunt(0, 1, 1, 0, -1, 1).args[0], Rational)
    assert tn(gaunt(10, 10, 12, 9, 3, -12, prec=64), Rational(-98, 62031) * sqrt(6279) / sqrt(pi))

    def gaunt_ref(l1, l2, l3, m1, m2, m3):
        return sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4 * pi)) * wigner_3j(l1, l2, l3, 0, 0, 0) * wigner_3j(l1, l2, l3, m1, m2, m3)
    threshold = 1e-10
    l_max = 3
    l3_max = 24
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            for l3 in range(l3_max + 1):
                for m1 in range(-l1, l1 + 1):
                    for m2 in range(-l2, l2 + 1):
                        for m3 in range(-l3, l3 + 1):
                            args = (l1, l2, l3, m1, m2, m3)
                            g = gaunt(*args)
                            g0 = gaunt_ref(*args)
                            assert abs(g - g0) < threshold
                            if m1 + m2 + m3 != 0:
                                assert abs(g) < threshold
                            if (l1 + l2 + l3) % 2:
                                assert abs(g) < threshold
    assert gaunt(1, 1, 0, 0, 2, -2) is S.Zero