from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex
from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
def test_FracElement_as_expr():
    F, x, y, z = field('x,y,z', ZZ)
    f = (3 * x ** 2 * y - x * y * z) / (7 * z ** 3 + 1)
    X, Y, Z = F.symbols
    g = (3 * X ** 2 * Y - X * Y * Z) / (7 * Z ** 3 + 1)
    assert f != g
    assert f.as_expr() == g
    X, Y, Z = symbols('x,y,z')
    g = (3 * X ** 2 * Y - X * Y * Z) / (7 * Z ** 3 + 1)
    assert f != g
    assert f.as_expr(X, Y, Z) == g
    raises(ValueError, lambda: f.as_expr(X))