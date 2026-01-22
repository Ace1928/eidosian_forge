from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dup_LC():
    assert dup_LC([], ZZ) == 0
    assert dup_LC([2, 3, 4, 5], ZZ) == 2