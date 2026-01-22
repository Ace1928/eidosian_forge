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
def test_PolyElement_pdiv():
    _, x, y = ring('x,y', ZZ)
    f, g = (x ** 2 - y ** 2, x - y)
    q, r = (x + y, 0)
    assert f.pdiv(g) == (q, r)
    assert f.prem(g) == r
    assert f.pquo(g) == q
    assert f.pexquo(g) == q