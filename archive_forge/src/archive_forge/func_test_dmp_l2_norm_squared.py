from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dmp_l2_norm_squared():
    assert dmp_l2_norm_squared([[[]]], 2, ZZ) == 0
    assert dmp_l2_norm_squared([[[1]]], 2, ZZ) == 1
    assert dmp_l2_norm_squared(f_0, 2, ZZ) == 111