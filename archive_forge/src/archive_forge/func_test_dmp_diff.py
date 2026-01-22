from sympy.polys.densebasic import (
from sympy.polys.densearith import dmp_mul_ground
from sympy.polys.densetools import (
from sympy.polys.polyclasses import ANP
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ, EX
from sympy.polys.rings import ring
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.functions.elementary.trigonometric import sin
from sympy.abc import x
from sympy.testing.pytest import raises
def test_dmp_diff():
    assert dmp_diff([], 1, 0, ZZ) == []
    assert dmp_diff([[]], 1, 1, ZZ) == [[]]
    assert dmp_diff([[[]]], 1, 2, ZZ) == [[[]]]
    assert dmp_diff([[[1], [2]]], 1, 2, ZZ) == [[[]]]
    assert dmp_diff([[[1]], [[]]], 1, 2, ZZ) == [[[1]]]
    assert dmp_diff([[[3]], [[1]], [[]]], 1, 2, ZZ) == [[[6]], [[1]]]
    assert dmp_diff([1, -1, 0, 0, 2], 1, 0, ZZ) == dup_diff([1, -1, 0, 0, 2], 1, ZZ)
    assert dmp_diff(f_6, 0, 3, ZZ) == f_6
    assert dmp_diff(f_6, 1, 3, ZZ) == [[[[8460]], [[]]], [[[135, 0, 0], [], [], [-135, 0, 0]]], [[[]]], [[[-423]], [[-47]], [[]], [[141], [], [94, 0], []], [[]]]]
    assert dmp_diff(f_6, 2, 3, ZZ) == dmp_diff(dmp_diff(f_6, 1, 3, ZZ), 1, 3, ZZ)
    assert dmp_diff(f_6, 3, 3, ZZ) == dmp_diff(dmp_diff(dmp_diff(f_6, 1, 3, ZZ), 1, 3, ZZ), 1, 3, ZZ)
    K = FF(23)
    F_6 = dmp_normal(f_6, 3, K)
    assert dmp_diff(F_6, 0, 3, K) == F_6
    assert dmp_diff(F_6, 1, 3, K) == dmp_diff(F_6, 1, 3, K)
    assert dmp_diff(F_6, 2, 3, K) == dmp_diff(dmp_diff(F_6, 1, 3, K), 1, 3, K)
    assert dmp_diff(F_6, 3, 3, K) == dmp_diff(dmp_diff(dmp_diff(F_6, 1, 3, K), 1, 3, K), 1, 3, K)