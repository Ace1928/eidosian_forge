from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_add_ground():
    f = ZZ.map([1, 2, 3, 4])
    g = ZZ.map([1, 2, 3, 8])
    assert dup_add_ground(f, ZZ(4), ZZ) == g