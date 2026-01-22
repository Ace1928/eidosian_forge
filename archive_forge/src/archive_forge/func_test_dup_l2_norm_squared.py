from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_l2_norm_squared():
    assert dup_l2_norm_squared([], ZZ) == 0
    assert dup_l2_norm_squared([1], ZZ) == 1
    assert dup_l2_norm_squared([1, 4, 2, 3], ZZ) == 30