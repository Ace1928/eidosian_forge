from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex
from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
@XFAIL
def test_FracElement___call__():
    F, x, y, z = field('x,y,z', ZZ)
    f = (x ** 2 + 3 * y) / z
    r = f(1, 1, 1)
    assert r == 4 and (not isinstance(r, FracElement))
    raises(ZeroDivisionError, lambda: f(1, 1, 0))