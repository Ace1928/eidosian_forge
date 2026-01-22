from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
from sympy.testing.pytest import raises
def test_dup_isolate_real_roots():
    R, x = ring('x', ZZ)
    assert R.dup_isolate_real_roots(0) == []
    assert R.dup_isolate_real_roots(3) == []
    assert R.dup_isolate_real_roots(5 * x) == [((0, 0), 1)]
    assert R.dup_isolate_real_roots(7 * x ** 4) == [((0, 0), 4)]
    assert R.dup_isolate_real_roots(x ** 2 + x) == [((-1, -1), 1), ((0, 0), 1)]
    assert R.dup_isolate_real_roots(x ** 2 - x) == [((0, 0), 1), ((1, 1), 1)]
    assert R.dup_isolate_real_roots(x ** 4 + x + 1) == []
    I = [((-2, -1), 1), ((1, 2), 1)]
    assert R.dup_isolate_real_roots(x ** 2 - 2) == I
    assert R.dup_isolate_real_roots(-x ** 2 + 2) == I
    f = 16 * x ** 14 - 96 * x ** 13 + 24 * x ** 12 + 936 * x ** 11 - 1599 * x ** 10 - 2880 * x ** 9 + 9196 * x ** 8 + 552 * x ** 7 - 21831 * x ** 6 + 13968 * x ** 5 + 21690 * x ** 4 - 26784 * x ** 3 - 2916 * x ** 2 + 15552 * x - 5832
    g = R.dup_sqf_part(f)
    assert R.dup_isolate_real_roots(f) == [((-QQ(2), -QQ(3, 2)), 2), ((-QQ(3, 2), -QQ(1, 1)), 3), ((QQ(1), QQ(3, 2)), 3), ((QQ(3, 2), QQ(3, 2)), 4), ((QQ(5, 3), QQ(2)), 2)]
    assert R.dup_isolate_real_roots_sqf(g) == [(-QQ(2), -QQ(3, 2)), (-QQ(3, 2), -QQ(1, 1)), (QQ(1), QQ(3, 2)), (QQ(3, 2), QQ(3, 2)), (QQ(3, 2), QQ(2))]
    assert R.dup_isolate_real_roots(g) == [((-QQ(2), -QQ(3, 2)), 1), ((-QQ(3, 2), -QQ(1, 1)), 1), ((QQ(1), QQ(3, 2)), 1), ((QQ(3, 2), QQ(3, 2)), 1), ((QQ(3, 2), QQ(2)), 1)]
    f = x - 1
    assert R.dup_isolate_real_roots(f, inf=2) == []
    assert R.dup_isolate_real_roots(f, sup=0) == []
    assert R.dup_isolate_real_roots(f) == [((1, 1), 1)]
    assert R.dup_isolate_real_roots(f, inf=1) == [((1, 1), 1)]
    assert R.dup_isolate_real_roots(f, sup=1) == [((1, 1), 1)]
    assert R.dup_isolate_real_roots(f, inf=1, sup=1) == [((1, 1), 1)]
    f = x ** 4 - 4 * x ** 2 + 4
    assert R.dup_isolate_real_roots(f, inf=QQ(7, 4)) == []
    assert R.dup_isolate_real_roots(f, inf=QQ(7, 5)) == [((QQ(7, 5), QQ(3, 2)), 2)]
    assert R.dup_isolate_real_roots(f, sup=QQ(7, 5)) == [((-2, -1), 2)]
    assert R.dup_isolate_real_roots(f, sup=QQ(7, 4)) == [((-2, -1), 2), ((1, QQ(3, 2)), 2)]
    assert R.dup_isolate_real_roots(f, sup=-QQ(7, 4)) == []
    assert R.dup_isolate_real_roots(f, sup=-QQ(7, 5)) == [((-QQ(3, 2), -QQ(7, 5)), 2)]
    assert R.dup_isolate_real_roots(f, inf=-QQ(7, 5)) == [((1, 2), 2)]
    assert R.dup_isolate_real_roots(f, inf=-QQ(7, 4)) == [((-QQ(3, 2), -1), 2), ((1, 2), 2)]
    I = [((-2, -1), 2), ((1, 2), 2)]
    assert R.dup_isolate_real_roots(f, inf=-2) == I
    assert R.dup_isolate_real_roots(f, sup=+2) == I
    assert R.dup_isolate_real_roots(f, inf=-2, sup=2) == I
    f = x ** 11 - 3 * x ** 10 - x ** 9 + 11 * x ** 8 - 8 * x ** 7 - 8 * x ** 6 + 12 * x ** 5 - 4 * x ** 4
    assert R.dup_isolate_real_roots(f, basis=False) == [((-2, -1), 2), ((0, 0), 4), ((1, 1), 3), ((1, 2), 2)]
    assert R.dup_isolate_real_roots(f, basis=True) == [((-2, -1), 2, [1, 0, -2]), ((0, 0), 4, [1, 0]), ((1, 1), 3, [1, -1]), ((1, 2), 2, [1, 0, -2])]
    f = x ** 45 - 45 * x ** 44 + 990 * x ** 43 - 1
    g = x ** 46 - 15180 * x ** 43 + 9366819 * x ** 40 - 53524680 * x ** 39 + 260932815 * x ** 38 - 1101716330 * x ** 37 + 4076350421 * x ** 36 - 13340783196 * x ** 35 + 38910617655 * x ** 34 - 101766230790 * x ** 33 + 239877544005 * x ** 32 - 511738760544 * x ** 31 + 991493848554 * x ** 30 - 1749695026860 * x ** 29 + 2818953098830 * x ** 28 - 4154246671960 * x ** 27 + 5608233007146 * x ** 26 - 6943526580276 * x ** 25 + 7890371113950 * x ** 24 - 8233430727600 * x ** 23 + 7890371113950 * x ** 22 - 6943526580276 * x ** 21 + 5608233007146 * x ** 20 - 4154246671960 * x ** 19 + 2818953098830 * x ** 18 - 1749695026860 * x ** 17 + 991493848554 * x ** 16 - 511738760544 * x ** 15 + 239877544005 * x ** 14 - 101766230790 * x ** 13 + 38910617655 * x ** 12 - 13340783196 * x ** 11 + 4076350421 * x ** 10 - 1101716330 * x ** 9 + 260932815 * x ** 8 - 53524680 * x ** 7 + 9366819 * x ** 6 - 1370754 * x ** 5 + 163185 * x ** 4 - 15180 * x ** 3 + 1035 * x ** 2 - 47 * x + 1
    assert R.dup_isolate_real_roots(f * g) == [((0, QQ(1, 2)), 1), ((QQ(2, 3), QQ(3, 4)), 1), ((QQ(3, 4), 1), 1), ((6, 7), 1), ((24, 25), 1)]
    R, x = ring('x', EX)
    raises(DomainError, lambda: R.dup_isolate_real_roots(x + 3))