from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dup_slice():
    f = [1, 2, 3, 4]
    assert dup_slice(f, 0, 0, ZZ) == []
    assert dup_slice(f, 0, 1, ZZ) == [4]
    assert dup_slice(f, 0, 2, ZZ) == [3, 4]
    assert dup_slice(f, 0, 3, ZZ) == [2, 3, 4]
    assert dup_slice(f, 0, 4, ZZ) == [1, 2, 3, 4]
    assert dup_slice(f, 0, 4, ZZ) == f
    assert dup_slice(f, 0, 9, ZZ) == f
    assert dup_slice(f, 1, 0, ZZ) == []
    assert dup_slice(f, 1, 1, ZZ) == []
    assert dup_slice(f, 1, 2, ZZ) == [3, 0]
    assert dup_slice(f, 1, 3, ZZ) == [2, 3, 0]
    assert dup_slice(f, 1, 4, ZZ) == [1, 2, 3, 0]
    assert dup_slice([1, 2], 0, 3, ZZ) == [1, 2]