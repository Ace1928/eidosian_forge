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
def test_PolyElement_drop():
    R, x, y, z = ring('x,y,z', ZZ)
    assert R(1).drop(0).ring == PolyRing('y,z', ZZ, lex)
    assert R(1).drop(0).drop(0).ring == PolyRing('z', ZZ, lex)
    assert isinstance(R(1).drop(0).drop(0).drop(0), R.dtype) is False
    raises(ValueError, lambda: z.drop(0).drop(0).drop(0))
    raises(ValueError, lambda: x.drop(0))