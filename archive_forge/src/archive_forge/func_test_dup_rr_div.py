from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_rr_div():
    raises(ZeroDivisionError, lambda: dup_rr_div([1, 2, 3], [], ZZ))
    f = dup_normal([3, 1, 1, 5], ZZ)
    g = dup_normal([5, -3, 1], ZZ)
    q, r = ([], f)
    assert dup_rr_div(f, g, ZZ) == (q, r)