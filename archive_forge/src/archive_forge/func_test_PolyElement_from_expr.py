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
def test_PolyElement_from_expr():
    x, y, z = symbols('x,y,z')
    R, X, Y, Z = ring((x, y, z), ZZ)
    f = R.from_expr(1)
    assert f == 1 and isinstance(f, R.dtype)
    f = R.from_expr(x)
    assert f == X and isinstance(f, R.dtype)
    f = R.from_expr(x * y * z)
    assert f == X * Y * Z and isinstance(f, R.dtype)
    f = R.from_expr(x * y * z + x * y + x)
    assert f == X * Y * Z + X * Y + X and isinstance(f, R.dtype)
    f = R.from_expr(x ** 3 * y * z + x ** 2 * y ** 7 + 1)
    assert f == X ** 3 * Y * Z + X ** 2 * Y ** 7 + 1 and isinstance(f, R.dtype)
    r, F = sring([exp(2)])
    f = r.from_expr(exp(2))
    assert f == F[0] and isinstance(f, r.dtype)
    raises(ValueError, lambda: R.from_expr(1 / x))
    raises(ValueError, lambda: R.from_expr(2 ** x))
    raises(ValueError, lambda: R.from_expr(7 * x + sqrt(2)))
    R, = ring('', ZZ)
    f = R.from_expr(1)
    assert f == 1 and isinstance(f, R.dtype)