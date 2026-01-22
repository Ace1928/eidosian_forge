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
def test_dup_clear_denoms():
    assert dup_clear_denoms([], QQ, ZZ) == (ZZ(1), [])
    assert dup_clear_denoms([QQ(1)], QQ, ZZ) == (ZZ(1), [QQ(1)])
    assert dup_clear_denoms([QQ(7)], QQ, ZZ) == (ZZ(1), [QQ(7)])
    assert dup_clear_denoms([QQ(7, 3)], QQ) == (ZZ(3), [QQ(7)])
    assert dup_clear_denoms([QQ(7, 3)], QQ, ZZ) == (ZZ(3), [QQ(7)])
    assert dup_clear_denoms([QQ(3), QQ(1), QQ(0)], QQ, ZZ) == (ZZ(1), [QQ(3), QQ(1), QQ(0)])
    assert dup_clear_denoms([QQ(1), QQ(1, 2), QQ(0)], QQ, ZZ) == (ZZ(2), [QQ(2), QQ(1), QQ(0)])
    assert dup_clear_denoms([QQ(3), QQ(1), QQ(0)], QQ, ZZ, convert=True) == (ZZ(1), [ZZ(3), ZZ(1), ZZ(0)])
    assert dup_clear_denoms([QQ(1), QQ(1, 2), QQ(0)], QQ, ZZ, convert=True) == (ZZ(2), [ZZ(2), ZZ(1), ZZ(0)])
    assert dup_clear_denoms([EX(S(3) / 2), EX(S(9) / 4)], EX) == (EX(4), [EX(6), EX(9)])
    assert dup_clear_denoms([EX(7)], EX) == (EX(1), [EX(7)])
    assert dup_clear_denoms([EX(sin(x) / x), EX(0)], EX) == (EX(x), [EX(sin(x)), EX(0)])