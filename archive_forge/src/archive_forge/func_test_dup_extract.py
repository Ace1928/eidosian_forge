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
def test_dup_extract():
    f = dup_normal([2930944, 0, 2198208, 0, 549552, 0, 45796], ZZ)
    g = dup_normal([17585664, 0, 8792832, 0, 1099104, 0], ZZ)
    F = dup_normal([64, 0, 48, 0, 12, 0, 1], ZZ)
    G = dup_normal([384, 0, 192, 0, 24, 0], ZZ)
    assert dup_extract(f, g, ZZ) == (45796, F, G)