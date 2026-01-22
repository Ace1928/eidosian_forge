from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex
from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
def test_FracElement_evaluate():
    F, x, y, z = field('x,y,z', ZZ)
    Fyz = field('y,z', ZZ)[0]
    f = (x ** 2 + 3 * y) / z
    assert f.evaluate(x, 0) == 3 * Fyz.y / Fyz.z
    raises(ZeroDivisionError, lambda: f.evaluate(z, 0))