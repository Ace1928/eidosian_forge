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
def test_dup_zz_i_factor():
    R, x = ring('x', ZZ_I)
    i = ZZ_I(0, 1)
    assert R.dup_zz_i_factor(x ** 2 - 2) == (ZZ_I(1, 0), [(x ** 2 - 2, 1)])
    assert R.dup_zz_i_factor(x ** 2 - 1) == (ZZ_I(1, 0), [(x - 1, 1), (x + 1, 1)])
    assert R.dup_zz_i_factor(x ** 2 + 1) == (ZZ_I(1, 0), [(x - i, 1), (x + i, 1)])
    assert R.dup_zz_i_factor(x ** 2 + 4) == (ZZ_I(1, 0), [(x - 2 * i, 1), (x + 2 * i, 1)])
    assert R.dup_zz_i_factor(x ** 2 + 2 * x + 1) == (ZZ_I(1, 0), [(x + 1, 2)])
    assert R.dup_zz_i_factor(x ** 2 + 2 * i * x - 1) == (ZZ_I(1, 0), [(x + i, 2)])
    f = 8192 * x ** 2 + x * (22656 + 175232 * i) - 921416 + 242313 * i
    assert R.dup_zz_i_factor(f) == (ZZ_I(0, 1), [((64 - 64 * i) * x + (773 + 596 * i), 2)])