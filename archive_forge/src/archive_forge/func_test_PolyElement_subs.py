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
def test_PolyElement_subs():
    R, x = ring('x', ZZ)
    f = x ** 3 + 4 * x ** 2 + 2 * x + 3
    r = f.subs(x, 0)
    assert r == 3 and isinstance(r, R.dtype)
    raises(CoercionFailed, lambda: f.subs(x, QQ(1, 7)))
    R, x, y, z = ring('x,y,z', ZZ)
    f = x ** 3 + 4 * x ** 2 + 2 * x + 3
    r = f.subs(x, 0)
    assert r == 3 and isinstance(r, R.dtype)
    r = f.subs([(x, 0), (y, 0)])
    assert r == 3 and isinstance(r, R.dtype)
    raises(CoercionFailed, lambda: f.subs([(x, 1), (y, QQ(1, 7))]))
    raises(CoercionFailed, lambda: f.subs([(x, QQ(1, 7)), (y, 1)]))
    raises(CoercionFailed, lambda: f.subs([(x, QQ(1, 7)), (y, QQ(1, 7))]))