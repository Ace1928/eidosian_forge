from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_exclude():
    assert dmp_exclude([[[]]], 2, ZZ) == ([], [[[]]], 2)
    assert dmp_exclude([[[7]]], 2, ZZ) == ([], [[[7]]], 2)
    assert dmp_exclude([1, 2, 3], 0, ZZ) == ([], [1, 2, 3], 0)
    assert dmp_exclude([[1], [2, 3]], 1, ZZ) == ([], [[1], [2, 3]], 1)
    assert dmp_exclude([[1, 2, 3]], 1, ZZ) == ([0], [1, 2, 3], 0)
    assert dmp_exclude([[1], [2], [3]], 1, ZZ) == ([1], [1, 2, 3], 0)
    assert dmp_exclude([[[1, 2, 3]]], 2, ZZ) == ([0, 1], [1, 2, 3], 0)
    assert dmp_exclude([[[1]], [[2]], [[3]]], 2, ZZ) == ([1, 2], [1, 2, 3], 0)