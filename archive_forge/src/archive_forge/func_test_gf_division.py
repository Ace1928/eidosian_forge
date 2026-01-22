from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_division():
    raises(ZeroDivisionError, lambda: gf_div([1, 2, 3], [], 11, ZZ))
    raises(ZeroDivisionError, lambda: gf_rem([1, 2, 3], [], 11, ZZ))
    raises(ZeroDivisionError, lambda: gf_quo([1, 2, 3], [], 11, ZZ))
    raises(ZeroDivisionError, lambda: gf_quo([1, 2, 3], [], 11, ZZ))
    assert gf_div([1], [1, 2, 3], 7, ZZ) == ([], [1])
    assert gf_rem([1], [1, 2, 3], 7, ZZ) == [1]
    assert gf_quo([1], [1, 2, 3], 7, ZZ) == []
    f = ZZ.map([5, 4, 3, 2, 1, 0])
    g = ZZ.map([1, 2, 3])
    q = [5, 1, 0, 6]
    r = [3, 3]
    assert gf_div(f, g, 7, ZZ) == (q, r)
    assert gf_rem(f, g, 7, ZZ) == r
    assert gf_quo(f, g, 7, ZZ) == q
    raises(ExactQuotientFailed, lambda: gf_exquo(f, g, 7, ZZ))
    f = ZZ.map([5, 4, 3, 2, 1, 0])
    g = ZZ.map([1, 2, 3, 0])
    q = [5, 1, 0]
    r = [6, 1, 0]
    assert gf_div(f, g, 7, ZZ) == (q, r)
    assert gf_rem(f, g, 7, ZZ) == r
    assert gf_quo(f, g, 7, ZZ) == q
    raises(ExactQuotientFailed, lambda: gf_exquo(f, g, 7, ZZ))
    assert gf_quo(ZZ.map([1, 2, 1]), ZZ.map([1, 1]), 11, ZZ) == [1, 1]