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
def test_PolyElement_degree():
    R, x, y, z = ring('x,y,z', ZZ)
    assert R(0).degree() is -oo
    assert R(1).degree() == 0
    assert (x + 1).degree() == 1
    assert (2 * y ** 3 + z).degree() == 0
    assert (x * y ** 3 + z).degree() == 1
    assert (x ** 5 * y ** 3 + z).degree() == 5
    assert R(0).degree(x) is -oo
    assert R(1).degree(x) == 0
    assert (x + 1).degree(x) == 1
    assert (2 * y ** 3 + z).degree(x) == 0
    assert (x * y ** 3 + z).degree(x) == 1
    assert (7 * x ** 5 * y ** 3 + z).degree(x) == 5
    assert R(0).degree(y) is -oo
    assert R(1).degree(y) == 0
    assert (x + 1).degree(y) == 0
    assert (2 * y ** 3 + z).degree(y) == 3
    assert (x * y ** 3 + z).degree(y) == 3
    assert (7 * x ** 5 * y ** 3 + z).degree(y) == 3
    assert R(0).degree(z) is -oo
    assert R(1).degree(z) == 0
    assert (x + 1).degree(z) == 0
    assert (2 * y ** 3 + z).degree(z) == 1
    assert (x * y ** 3 + z).degree(z) == 1
    assert (7 * x ** 5 * y ** 3 + z).degree(z) == 1
    R, = ring('', ZZ)
    assert R(0).degree() is -oo
    assert R(1).degree() == 0