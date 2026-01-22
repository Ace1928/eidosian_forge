from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dup_deflate():
    assert dup_deflate([], ZZ) == (1, [])
    assert dup_deflate([2], ZZ) == (1, [2])
    assert dup_deflate([1, 2, 3], ZZ) == (1, [1, 2, 3])
    assert dup_deflate([1, 0, 2, 0, 3], ZZ) == (2, [1, 2, 3])
    assert dup_deflate(dup_from_raw_dict({7: 1, 1: 1}, ZZ), ZZ) == (1, [1, 0, 0, 0, 0, 0, 1, 0])
    assert dup_deflate(dup_from_raw_dict({7: 1, 0: 1}, ZZ), ZZ) == (7, [1, 1])
    assert dup_deflate(dup_from_raw_dict({7: 1, 3: 1}, ZZ), ZZ) == (1, [1, 0, 0, 0, 1, 0, 0, 0])
    assert dup_deflate(dup_from_raw_dict({7: 1, 4: 1}, ZZ), ZZ) == (1, [1, 0, 0, 1, 0, 0, 0, 0])
    assert dup_deflate(dup_from_raw_dict({8: 1, 4: 1}, ZZ), ZZ) == (4, [1, 1, 0])
    assert dup_deflate(dup_from_raw_dict({8: 1}, ZZ), ZZ) == (8, [1, 0])
    assert dup_deflate(dup_from_raw_dict({7: 1}, ZZ), ZZ) == (7, [1, 0])
    assert dup_deflate(dup_from_raw_dict({1: 1}, ZZ), ZZ) == (1, [1, 0])