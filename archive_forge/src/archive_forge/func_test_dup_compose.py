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
def test_dup_compose():
    assert dup_compose([], [], ZZ) == []
    assert dup_compose([], [1], ZZ) == []
    assert dup_compose([], [1, 2], ZZ) == []
    assert dup_compose([1], [], ZZ) == [1]
    assert dup_compose([1, 2, 0], [], ZZ) == []
    assert dup_compose([1, 2, 1], [], ZZ) == [1]
    assert dup_compose([1, 2, 1], [1], ZZ) == [4]
    assert dup_compose([1, 2, 1], [7], ZZ) == [64]
    assert dup_compose([1, 2, 1], [1, -1], ZZ) == [1, 0, 0]
    assert dup_compose([1, 2, 1], [1, 1], ZZ) == [1, 4, 4]
    assert dup_compose([1, 2, 1], [1, 2, 1], ZZ) == [1, 4, 8, 8, 4]