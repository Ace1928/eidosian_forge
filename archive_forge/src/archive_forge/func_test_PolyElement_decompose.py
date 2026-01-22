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
def test_PolyElement_decompose():
    _, x = ring('x', ZZ)
    f = x ** 12 + 20 * x ** 10 + 150 * x ** 8 + 500 * x ** 6 + 625 * x ** 4 - 2 * x ** 3 - 10 * x + 9
    g = x ** 4 - 2 * x + 9
    h = x ** 3 + 5 * x
    assert g.compose(x, h) == f
    assert f.decompose() == [g, h]