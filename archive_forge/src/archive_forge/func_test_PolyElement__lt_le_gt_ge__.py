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
def test_PolyElement__lt_le_gt_ge__():
    R, x, y = ring('x,y', ZZ)
    assert R(1) < x < x ** 2 < x ** 3
    assert R(1) <= x <= x ** 2 <= x ** 3
    assert x ** 3 > x ** 2 > x > R(1)
    assert x ** 3 >= x ** 2 >= x >= R(1)