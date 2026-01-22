from functools import reduce
from operator import add, mul
from sympy.polys.rings import ring, xring, sring, PolyRing, PolyElement
from sympy.polys.fields import field, FracField
from sympy.polys.domains import ZZ, QQ, RR, FF, EX
from sympy.polys.orderings import lex, grlex
from sympy.polys.polyerrors import GeneratorsError, \
from sympy.testing.pytest import raises
from sympy.core import Symbol, symbols
from sympy.core.singleton import S
from sympy.core.numbers import (oo, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
def test_PolyElement___truediv__():
    R, x, y, z = ring('x,y,z', ZZ)
    assert (2 * x ** 2 - 4) / 2 == x ** 2 - 2
    assert (2 * x ** 2 - 3) / 2 == x ** 2
    assert (x ** 2 - 1).quo(x) == x
    assert (x ** 2 - x).quo(x) == x - 1
    assert (x ** 2 - 1) / x == x - x ** (-1)
    assert (x ** 2 - x) / x == x - 1
    assert (x ** 2 - 1) / (2 * x) == x / 2 - x ** (-1) / 2
    assert (x ** 2 - 1).quo(2 * x) == 0
    assert (x ** 2 - x) / (x - 1) == (x ** 2 - x).quo(x - 1) == x
    R, x, y, z = ring('x,y,z', ZZ)
    assert len((x ** 2 / 3 + y ** 3 / 4 + z ** 4 / 5).terms()) == 0
    R, x, y, z = ring('x,y,z', QQ)
    assert len((x ** 2 / 3 + y ** 3 / 4 + z ** 4 / 5).terms()) == 3
    Rt, t = ring('t', ZZ)
    Ruv, u, v = ring('u,v', ZZ)
    Rxyz, x, y, z = ring('x,y,z', Ruv)
    assert dict((u ** 2 * x + u) / u) == {(1, 0, 0): u, (0, 0, 0): 1}
    raises(TypeError, lambda: u / (u ** 2 * x + u))
    raises(TypeError, lambda: t / x)
    raises(TypeError, lambda: x / t)
    raises(TypeError, lambda: t / u)
    raises(TypeError, lambda: u / t)
    R, x = ring('x', ZZ)
    f, g = (x ** 2 + 2 * x + 3, R(0))
    raises(ZeroDivisionError, lambda: f.div(g))
    raises(ZeroDivisionError, lambda: divmod(f, g))
    raises(ZeroDivisionError, lambda: f.rem(g))
    raises(ZeroDivisionError, lambda: f % g)
    raises(ZeroDivisionError, lambda: f.quo(g))
    raises(ZeroDivisionError, lambda: f / g)
    raises(ZeroDivisionError, lambda: f.exquo(g))
    R, x, y = ring('x,y', ZZ)
    f, g = (x * y + 2 * x + 3, R(0))
    raises(ZeroDivisionError, lambda: f.div(g))
    raises(ZeroDivisionError, lambda: divmod(f, g))
    raises(ZeroDivisionError, lambda: f.rem(g))
    raises(ZeroDivisionError, lambda: f % g)
    raises(ZeroDivisionError, lambda: f.quo(g))
    raises(ZeroDivisionError, lambda: f / g)
    raises(ZeroDivisionError, lambda: f.exquo(g))
    R, x = ring('x', ZZ)
    f, g = (x ** 2 + 1, 2 * x - 4)
    q, r = (R(0), x ** 2 + 1)
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    f, g = (3 * x ** 3 + x ** 2 + x + 5, 5 * x ** 2 - 3 * x + 1)
    q, r = (R(0), f)
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    f, g = (5 * x ** 4 + 4 * x ** 3 + 3 * x ** 2 + 2 * x + 1, x ** 2 + 2 * x + 3)
    q, r = (5 * x ** 2 - 6 * x, 20 * x + 1)
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    f, g = (5 * x ** 5 + 4 * x ** 4 + 3 * x ** 3 + 2 * x ** 2 + x, x ** 4 + 2 * x ** 3 + 9)
    q, r = (5 * x - 6, 15 * x ** 3 + 2 * x ** 2 - 44 * x + 54)
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    R, x = ring('x', QQ)
    f, g = (x ** 2 + 1, 2 * x - 4)
    q, r = (x / 2 + 1, R(5))
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    f, g = (3 * x ** 3 + x ** 2 + x + 5, 5 * x ** 2 - 3 * x + 1)
    q, r = (QQ(3, 5) * x + QQ(14, 25), QQ(52, 25) * x + QQ(111, 25))
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    R, x, y = ring('x,y', ZZ)
    f, g = (x ** 2 - y ** 2, x - y)
    q, r = (x + y, R(0))
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    assert f.exquo(g) == q
    f, g = (x ** 2 + y ** 2, x - y)
    q, r = (x + y, 2 * y ** 2)
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    f, g = (x ** 2 + y ** 2, -x + y)
    q, r = (-x - y, 2 * y ** 2)
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    f, g = (x ** 2 + y ** 2, 2 * x - 2 * y)
    q, r = (R(0), f)
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    R, x, y = ring('x,y', QQ)
    f, g = (x ** 2 - y ** 2, x - y)
    q, r = (x + y, R(0))
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    assert f.exquo(g) == q
    f, g = (x ** 2 + y ** 2, x - y)
    q, r = (x + y, 2 * y ** 2)
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    f, g = (x ** 2 + y ** 2, -x + y)
    q, r = (-x - y, 2 * y ** 2)
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    f, g = (x ** 2 + y ** 2, 2 * x - 2 * y)
    q, r = (x / 2 + y / 2, 2 * y ** 2)
    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == f / g == q
    raises(ExactQuotientFailed, lambda: f.exquo(g))