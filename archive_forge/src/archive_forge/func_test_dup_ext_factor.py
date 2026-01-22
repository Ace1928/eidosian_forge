from sympy.polys.rings import ring, xring
from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, RR, EX
from sympy.polys import polyconfig as config
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyclasses import ANP
from sympy.polys.specialpolys import f_polys, w_polys
from sympy.core.numbers import I
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises, XFAIL
def test_dup_ext_factor():
    R, x = ring('x', QQ.algebraic_field(I))

    def anp(element):
        return ANP(element, [QQ(1), QQ(0), QQ(1)], QQ)
    assert R.dup_ext_factor(0) == (anp([]), [])
    f = anp([QQ(1)]) * x + anp([QQ(1)])
    assert R.dup_ext_factor(f) == (anp([QQ(1)]), [(f, 1)])
    g = anp([QQ(2)]) * x + anp([QQ(2)])
    assert R.dup_ext_factor(g) == (anp([QQ(2)]), [(f, 1)])
    f = anp([QQ(7)]) * x ** 4 + anp([QQ(1, 1)])
    g = anp([QQ(1)]) * x ** 4 + anp([QQ(1, 7)])
    assert R.dup_ext_factor(f) == (anp([QQ(7)]), [(g, 1)])
    f = anp([QQ(1)]) * x ** 4 + anp([QQ(1)])
    assert R.dup_ext_factor(f) == (anp([QQ(1, 1)]), [(anp([QQ(1)]) * x ** 2 + anp([QQ(-1), QQ(0)]), 1), (anp([QQ(1)]) * x ** 2 + anp([QQ(1), QQ(0)]), 1)])
    f = anp([QQ(4, 1)]) * x ** 2 + anp([QQ(9, 1)])
    assert R.dup_ext_factor(f) == (anp([QQ(4, 1)]), [(anp([QQ(1, 1)]) * x + anp([-QQ(3, 2), QQ(0, 1)]), 1), (anp([QQ(1, 1)]) * x + anp([QQ(3, 2), QQ(0, 1)]), 1)])
    f = anp([QQ(4, 1)]) * x ** 4 + anp([QQ(8, 1)]) * x ** 3 + anp([QQ(77, 1)]) * x ** 2 + anp([QQ(18, 1)]) * x + anp([QQ(153, 1)])
    assert R.dup_ext_factor(f) == (anp([QQ(4, 1)]), [(anp([QQ(1, 1)]) * x + anp([-QQ(4, 1), QQ(1, 1)]), 1), (anp([QQ(1, 1)]) * x + anp([-QQ(3, 2), QQ(0, 1)]), 1), (anp([QQ(1, 1)]) * x + anp([QQ(3, 2), QQ(0, 1)]), 1), (anp([QQ(1, 1)]) * x + anp([QQ(4, 1), QQ(1, 1)]), 1)])
    R, x = ring('x', QQ.algebraic_field(sqrt(2)))

    def anp(element):
        return ANP(element, [QQ(1), QQ(0), QQ(-2)], QQ)
    f = anp([QQ(1)]) * x ** 4 + anp([QQ(1, 1)])
    assert R.dup_ext_factor(f) == (anp([QQ(1)]), [(anp([QQ(1)]) * x ** 2 + anp([QQ(-1), QQ(0)]) * x + anp([QQ(1)]), 1), (anp([QQ(1)]) * x ** 2 + anp([QQ(1), QQ(0)]) * x + anp([QQ(1)]), 1)])
    f = anp([QQ(1, 1)]) * x ** 2 + anp([QQ(2), QQ(0)]) * x + anp([QQ(2, 1)])
    assert R.dup_ext_factor(f) == (anp([QQ(1, 1)]), [(anp([1]) * x + anp([1, 0]), 2)])
    assert R.dup_ext_factor(f ** 3) == (anp([QQ(1, 1)]), [(anp([1]) * x + anp([1, 0]), 6)])
    f *= anp([QQ(2, 1)])
    assert R.dup_ext_factor(f) == (anp([QQ(2, 1)]), [(anp([1]) * x + anp([1, 0]), 2)])
    assert R.dup_ext_factor(f ** 3) == (anp([QQ(8, 1)]), [(anp([1]) * x + anp([1, 0]), 6)])