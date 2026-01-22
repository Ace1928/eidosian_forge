from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_swap():
    f = dmp_normal([[1, 0, 0], [], [1, 0], [], [1]], 1, ZZ)
    g = dmp_normal([[1, 0, 0, 0, 0], [1, 0, 0], [1]], 1, ZZ)
    assert dmp_swap(f, 1, 1, 1, ZZ) == f
    assert dmp_swap(f, 0, 1, 1, ZZ) == g
    assert dmp_swap(g, 0, 1, 1, ZZ) == f
    raises(IndexError, lambda: dmp_swap(f, -1, -7, 1, ZZ))