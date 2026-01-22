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
def test_PolyRing_drop():
    R, x, y, z = ring('x,y,z', ZZ)
    assert R.drop(x) == PolyRing('y,z', ZZ, lex)
    assert R.drop(y) == PolyRing('x,z', ZZ, lex)
    assert R.drop(z) == PolyRing('x,y', ZZ, lex)
    assert R.drop(0) == PolyRing('y,z', ZZ, lex)
    assert R.drop(0).drop(0) == PolyRing('z', ZZ, lex)
    assert R.drop(0).drop(0).drop(0) == ZZ
    assert R.drop(1) == PolyRing('x,z', ZZ, lex)
    assert R.drop(2) == PolyRing('x,y', ZZ, lex)
    assert R.drop(2).drop(1) == PolyRing('x', ZZ, lex)
    assert R.drop(2).drop(1).drop(0) == ZZ
    raises(ValueError, lambda: R.drop(3))
    raises(ValueError, lambda: R.drop(x).drop(y))