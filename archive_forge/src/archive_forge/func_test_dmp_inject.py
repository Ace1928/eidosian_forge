from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_inject():
    R, x, y = ring('x,y', ZZ)
    K = R.to_domain()
    assert dmp_inject([], 0, K) == ([[[]]], 2)
    assert dmp_inject([[]], 1, K) == ([[[[]]]], 3)
    assert dmp_inject([R(1)], 0, K) == ([[[1]]], 2)
    assert dmp_inject([[R(1)]], 1, K) == ([[[[1]]]], 3)
    assert dmp_inject([R(1), 2 * x + 3 * y + 4], 0, K) == ([[[1]], [[2], [3, 4]]], 2)
    f = [3 * x ** 2 + 7 * x * y + 5 * y ** 2, 2 * x, R(0), x * y ** 2 + 11]
    g = [[[3], [7, 0], [5, 0, 0]], [[2], []], [[]], [[1, 0, 0], [11]]]
    assert dmp_inject(f, 0, K) == (g, 2)