from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex
from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
def test_FracElement___pow__():
    F, x, y = field('x,y', QQ)
    f, g = (1 / x, 1 / y)
    assert f ** 3 == 1 / x ** 3
    assert g ** 3 == 1 / y ** 3
    assert (f * g) ** 3 == 1 / (x ** 3 * y ** 3)
    assert (f * g) ** (-3) == (x * y) ** 3
    raises(ZeroDivisionError, lambda: (x - x) ** (-3))