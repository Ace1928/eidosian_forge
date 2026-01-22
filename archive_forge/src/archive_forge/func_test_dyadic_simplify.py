from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.vector import (CoordSys3D, Vector, Dyadic,
def test_dyadic_simplify():
    x, y, z, k, n, m, w, f, s, A = symbols('x, y, z, k, n, m, w, f, s, A')
    N = CoordSys3D('N')
    dy = N.i | N.i
    test1 = (1 / x + 1 / y) * dy
    assert N.i & test1 & N.i != (x + y) / (x * y)
    test1 = test1.simplify()
    assert test1.simplify() == simplify(test1)
    assert N.i & test1 & N.i == (x + y) / (x * y)
    test2 = A ** 2 * s ** 4 / (4 * pi * k * m ** 3) * dy
    test2 = test2.simplify()
    assert N.i & test2 & N.i == A ** 2 * s ** 4 / (4 * pi * k * m ** 3)
    test3 = (4 + 4 * x - 2 * (2 + 2 * x)) / (2 + 2 * x) * dy
    test3 = test3.simplify()
    assert N.i & test3 & N.i == 0
    test4 = (-4 * x * y ** 2 - 2 * y ** 3 - 2 * x ** 2 * y) / (x + y) ** 2 * dy
    test4 = test4.simplify()
    assert N.i & test4 & N.i == -2 * y