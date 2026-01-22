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
def test_PolyRing_symmetric_poly():
    R, x, y, z, t = ring('x,y,z,t', ZZ)
    raises(ValueError, lambda: R.symmetric_poly(-1))
    raises(ValueError, lambda: R.symmetric_poly(5))
    assert R.symmetric_poly(0) == R.one
    assert R.symmetric_poly(1) == x + y + z + t
    assert R.symmetric_poly(2) == x * y + x * z + x * t + y * z + y * t + z * t
    assert R.symmetric_poly(3) == x * y * z + x * y * t + x * z * t + y * z * t
    assert R.symmetric_poly(4) == x * y * z * t