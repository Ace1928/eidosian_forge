from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex
from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
def test_FracElement_copy():
    F, x, y, z = field('x,y,z', ZZ)
    f = x * y / 3 * z
    g = f.copy()
    assert f == g
    g.numer[1, 1, 1] = 7
    assert f != g