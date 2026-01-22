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
def test_PolyElement_deflate():
    R, x = ring('x', ZZ)
    assert (2 * x ** 2).deflate(x ** 4 + 4 * x ** 2 + 1) == ((2,), [2 * x, x ** 2 + 4 * x + 1])
    R, x, y = ring('x,y', ZZ)
    assert R(0).deflate(R(0)) == ((1, 1), [0, 0])
    assert R(1).deflate(R(0)) == ((1, 1), [1, 0])
    assert R(1).deflate(R(2)) == ((1, 1), [1, 2])
    assert R(1).deflate(2 * y) == ((1, 1), [1, 2 * y])
    assert (2 * y).deflate(2 * y) == ((1, 1), [2 * y, 2 * y])
    assert R(2).deflate(2 * y ** 2) == ((1, 2), [2, 2 * y])
    assert (2 * y ** 2).deflate(2 * y ** 2) == ((1, 2), [2 * y, 2 * y])
    f = x ** 4 * y ** 2 + x ** 2 * y + 1
    g = x ** 2 * y ** 3 + x ** 2 * y + 1
    assert f.deflate(g) == ((2, 1), [x ** 2 * y ** 2 + x * y + 1, x * y ** 3 + x * y + 1])