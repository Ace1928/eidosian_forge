from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_lshift():
    assert dup_lshift([], 3, ZZ) == []
    assert dup_lshift([1], 3, ZZ) == [1, 0, 0, 0]