from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dup_from_to_dict():
    assert dup_from_raw_dict({}, ZZ) == []
    assert dup_from_dict({}, ZZ) == []
    assert dup_to_raw_dict([]) == {}
    assert dup_to_dict([]) == {}
    assert dup_to_raw_dict([], ZZ, zero=True) == {0: ZZ(0)}
    assert dup_to_dict([], ZZ, zero=True) == {(0,): ZZ(0)}
    f = [3, 0, 0, 2, 0, 0, 0, 0, 8]
    g = {8: 3, 5: 2, 0: 8}
    h = {(8,): 3, (5,): 2, (0,): 8}
    assert dup_from_raw_dict(g, ZZ) == f
    assert dup_from_dict(h, ZZ) == f
    assert dup_to_raw_dict(f) == g
    assert dup_to_dict(f) == h
    R, x, y = ring('x,y', ZZ)
    K = R.to_domain()
    f = [R(3), R(0), R(2), R(0), R(0), R(8)]
    g = {5: R(3), 3: R(2), 0: R(8)}
    h = {(5,): R(3), (3,): R(2), (0,): R(8)}
    assert dup_from_raw_dict(g, K) == f
    assert dup_from_dict(h, K) == f
    assert dup_to_raw_dict(f) == g
    assert dup_to_dict(f) == h