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
def test_PolyElement_tail_degree():
    R, x, y, z = ring('x,y,z', ZZ)
    assert R(0).tail_degree() is -oo
    assert R(1).tail_degree() == 0
    assert (x + 1).tail_degree() == 0
    assert (2 * y ** 3 + x ** 3 * z).tail_degree() == 0
    assert (x * y ** 3 + x ** 3 * z).tail_degree() == 1
    assert (x ** 5 * y ** 3 + x ** 3 * z).tail_degree() == 3
    assert R(0).tail_degree(x) is -oo
    assert R(1).tail_degree(x) == 0
    assert (x + 1).tail_degree(x) == 0
    assert (2 * y ** 3 + x ** 3 * z).tail_degree(x) == 0
    assert (x * y ** 3 + x ** 3 * z).tail_degree(x) == 1
    assert (7 * x ** 5 * y ** 3 + x ** 3 * z).tail_degree(x) == 3
    assert R(0).tail_degree(y) is -oo
    assert R(1).tail_degree(y) == 0
    assert (x + 1).tail_degree(y) == 0
    assert (2 * y ** 3 + x ** 3 * z).tail_degree(y) == 0
    assert (x * y ** 3 + x ** 3 * z).tail_degree(y) == 0
    assert (7 * x ** 5 * y ** 3 + x ** 3 * z).tail_degree(y) == 0
    assert R(0).tail_degree(z) is -oo
    assert R(1).tail_degree(z) == 0
    assert (x + 1).tail_degree(z) == 0
    assert (2 * y ** 3 + x ** 3 * z).tail_degree(z) == 0
    assert (x * y ** 3 + x ** 3 * z).tail_degree(z) == 0
    assert (7 * x ** 5 * y ** 3 + x ** 3 * z).tail_degree(z) == 0
    R, = ring('', ZZ)
    assert R(0).tail_degree() is -oo
    assert R(1).tail_degree() == 0