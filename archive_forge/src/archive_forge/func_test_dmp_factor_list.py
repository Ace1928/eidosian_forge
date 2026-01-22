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
def test_dmp_factor_list():
    R, x, y = ring('x,y', ZZ)
    assert R.dmp_factor_list(0) == (ZZ(0), [])
    assert R.dmp_factor_list(7) == (7, [])
    R, x, y = ring('x,y', QQ)
    assert R.dmp_factor_list(0) == (QQ(0), [])
    assert R.dmp_factor_list(QQ(1, 7)) == (QQ(1, 7), [])
    Rt, t = ring('t', ZZ)
    R, x, y = ring('x,y', Rt)
    assert R.dmp_factor_list(0) == (0, [])
    assert R.dmp_factor_list(7) == (ZZ(7), [])
    Rt, t = ring('t', QQ)
    R, x, y = ring('x,y', Rt)
    assert R.dmp_factor_list(0) == (0, [])
    assert R.dmp_factor_list(QQ(1, 7)) == (QQ(1, 7), [])
    R, x, y = ring('x,y', ZZ)
    assert R.dmp_factor_list_include(0) == [(0, 1)]
    assert R.dmp_factor_list_include(7) == [(7, 1)]
    R, X = xring('x:200', ZZ)
    f, g = (X[0] ** 2 + 2 * X[0] + 1, X[0] + 1)
    assert R.dmp_factor_list(f) == (1, [(g, 2)])
    f, g = (X[-1] ** 2 + 2 * X[-1] + 1, X[-1] + 1)
    assert R.dmp_factor_list(f) == (1, [(g, 2)])
    R, x = ring('x', ZZ)
    assert R.dmp_factor_list(x ** 2 + 2 * x + 1) == (1, [(x + 1, 2)])
    R, x = ring('x', QQ)
    assert R.dmp_factor_list(QQ(1, 2) * x ** 2 + x + QQ(1, 2)) == (QQ(1, 2), [(x + 1, 2)])
    R, x, y = ring('x,y', ZZ)
    assert R.dmp_factor_list(x ** 2 + 2 * x + 1) == (1, [(x + 1, 2)])
    R, x, y = ring('x,y', QQ)
    assert R.dmp_factor_list(QQ(1, 2) * x ** 2 + x + QQ(1, 2)) == (QQ(1, 2), [(x + 1, 2)])
    R, x, y = ring('x,y', ZZ)
    f = 4 * x ** 2 * y + 4 * x * y ** 2
    assert R.dmp_factor_list(f) == (4, [(y, 1), (x, 1), (x + y, 1)])
    assert R.dmp_factor_list_include(f) == [(4 * y, 1), (x, 1), (x + y, 1)]
    R, x, y = ring('x,y', QQ)
    f = QQ(1, 2) * x ** 2 * y + QQ(1, 2) * x * y ** 2
    assert R.dmp_factor_list(f) == (QQ(1, 2), [(y, 1), (x, 1), (x + y, 1)])
    R, x, y = ring('x,y', RR)
    f = 2.0 * x ** 2 - 8.0 * y ** 2
    assert R.dmp_factor_list(f) == (RR(8.0), [(0.5 * x - y, 1), (0.5 * x + y, 1)])
    f = 6.7225336055071 * x ** 2 * y ** 2 - 10.6463972754741 * x * y - 0.33469524022264
    coeff, factors = R.dmp_factor_list(f)
    assert coeff == RR(10.6463972754741)
    assert len(factors) == 1
    assert factors[0][0].max_norm() == RR(1.0)
    assert factors[0][1] == 1
    Rt, t = ring('t', ZZ)
    R, x, y = ring('x,y', Rt)
    f = 4 * t * x ** 2 + 4 * t ** 2 * x
    assert R.dmp_factor_list(f) == (4 * t, [(x, 1), (x + t, 1)])
    Rt, t = ring('t', QQ)
    R, x, y = ring('x,y', Rt)
    f = QQ(1, 2) * t * x ** 2 + QQ(1, 2) * t ** 2 * x
    assert R.dmp_factor_list(f) == (QQ(1, 2) * t, [(x, 1), (x + t, 1)])
    R, x, y = ring('x,y', FF(2))
    raises(NotImplementedError, lambda: R.dmp_factor_list(x ** 2 + y ** 2))
    R, x, y = ring('x,y', EX)
    raises(DomainError, lambda: R.dmp_factor_list(EX(sin(1))))