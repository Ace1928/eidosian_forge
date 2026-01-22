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
def test_dup_zz_cyclotomic_factor():
    R, x = ring('x', ZZ)
    assert R.dup_zz_cyclotomic_factor(0) is None
    assert R.dup_zz_cyclotomic_factor(1) is None
    assert R.dup_zz_cyclotomic_factor(2 * x ** 10 - 1) is None
    assert R.dup_zz_cyclotomic_factor(x ** 10 - 3) is None
    assert R.dup_zz_cyclotomic_factor(x ** 10 + x ** 5 - 1) is None
    assert R.dup_zz_cyclotomic_factor(x + 1) == [x + 1]
    assert R.dup_zz_cyclotomic_factor(x - 1) == [x - 1]
    assert R.dup_zz_cyclotomic_factor(x ** 2 + 1) == [x ** 2 + 1]
    assert R.dup_zz_cyclotomic_factor(x ** 2 - 1) == [x - 1, x + 1]
    assert R.dup_zz_cyclotomic_factor(x ** 27 + 1) == [x + 1, x ** 2 - x + 1, x ** 6 - x ** 3 + 1, x ** 18 - x ** 9 + 1]
    assert R.dup_zz_cyclotomic_factor(x ** 27 - 1) == [x - 1, x ** 2 + x + 1, x ** 6 + x ** 3 + 1, x ** 18 + x ** 9 + 1]