from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dup_random():
    f = dup_random(0, -10, 10, ZZ)
    assert dup_degree(f) == 0
    assert all((-10 <= c <= 10 for c in f))
    f = dup_random(1, -20, 20, ZZ)
    assert dup_degree(f) == 1
    assert all((-20 <= c <= 20 for c in f))
    f = dup_random(2, -30, 30, ZZ)
    assert dup_degree(f) == 2
    assert all((-30 <= c <= 30 for c in f))
    f = dup_random(3, -40, 40, ZZ)
    assert dup_degree(f) == 3
    assert all((-40 <= c <= 40 for c in f))