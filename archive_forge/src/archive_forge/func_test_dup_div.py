from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_div():
    f, g, q, r = ([5, 4, 3, 2, 1], [1, 2, 3], [5, -6, 0], [20, 1])
    assert dup_div(f, g, ZZ) == (q, r)
    assert dup_quo(f, g, ZZ) == q
    assert dup_rem(f, g, ZZ) == r
    raises(ExactQuotientFailed, lambda: dup_exquo(f, g, ZZ))
    f, g, q, r = ([5, 4, 3, 2, 1, 0], [1, 2, 0, 0, 9], [5, -6], [15, 2, -44, 54])
    assert dup_div(f, g, ZZ) == (q, r)
    assert dup_quo(f, g, ZZ) == q
    assert dup_rem(f, g, ZZ) == r
    raises(ExactQuotientFailed, lambda: dup_exquo(f, g, ZZ))