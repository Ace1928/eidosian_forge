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
@XFAIL
def test_dmp_zz_wang_fail():
    R, x, y, z = ring('x,y,z', ZZ)
    UV, _x = ring('x', ZZ)
    p = ZZ(nextprime(R.dmp_zz_mignotte_bound(w_1)))
    assert p == 6291469
    H_1 = [44 * x ** 2 + 42 * x + 1, 126 * x ** 2 - 9 * x + 28, 187 * x ** 2 - 23]
    H_2 = [-4 * x ** 2 * y - 12 * x ** 2 - 3 * x * y + 1, -9 * x ** 2 * y - 9 * x - 2 * y, x ** 2 * y ** 2 - 9 * x ** 2 + y - 9]
    H_3 = [-4 * x ** 2 * y - 12 * x ** 2 - 3 * x * y + 1, -9 * x ** 2 * y - 9 * x - 2 * y, x ** 2 * y ** 2 - 9 * x ** 2 + y - 9]
    c_1 = -70686 * x ** 5 - 5863 * x ** 4 - 17826 * x ** 3 + 2009 * x ** 2 + 5031 * x + 74
    c_2 = 9 * x ** 5 * y ** 4 + 12 * x ** 5 * y ** 3 - 45 * x ** 5 * y ** 2 - 108 * x ** 5 * y - 324 * x ** 5 + 18 * x ** 4 * y ** 3 - 216 * x ** 4 * y ** 2 - 810 * x ** 4 * y + 2 * x ** 3 * y ** 4 + 9 * x ** 3 * y ** 3 - 252 * x ** 3 * y ** 2 - 288 * x ** 3 * y - 945 * x ** 3 - 30 * x ** 2 * y ** 2 - 414 * x ** 2 * y + 2 * x * y ** 3 - 54 * x * y ** 2 - 3 * x * y + 81 * x + 12 * y
    c_3 = -36 * x ** 4 * y ** 2 - 108 * x ** 4 * y - 27 * x ** 3 * y ** 2 - 36 * x ** 3 * y - 108 * x ** 3 - 8 * x ** 2 * y ** 2 - 42 * x ** 2 * y - 6 * x * y ** 2 + 9 * x + 2 * y
    assert R.dmp_zz_diophantine(H_1, c_1, [], 5, p) == [-3 * x, -2, 1]
    assert R.dmp_zz_diophantine(H_2, c_2, [ZZ(-14)], 5, p) == [-x * y, -3 * x, -6]
    assert R.dmp_zz_diophantine(H_3, c_3, [ZZ(-14)], 5, p) == [0, 0, -1]