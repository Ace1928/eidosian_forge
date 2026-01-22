from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dup_strip():
    assert dup_strip([]) == []
    assert dup_strip([0]) == []
    assert dup_strip([0, 0, 0]) == []
    assert dup_strip([1]) == [1]
    assert dup_strip([0, 1]) == [1]
    assert dup_strip([0, 0, 0, 1]) == [1]
    assert dup_strip([1, 2, 0]) == [1, 2, 0]
    assert dup_strip([0, 1, 2, 0]) == [1, 2, 0]
    assert dup_strip([0, 0, 0, 1, 2, 0]) == [1, 2, 0]