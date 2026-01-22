from sympy.core.numbers import (Float, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.functions.elementary.trigonometric import tan
from sympy.geometry import (Circle, Ellipse, GeometryError, Point, Point2D,
from sympy.testing.pytest import raises, slow, warns
from sympy.core.random import verify_numerically
from sympy.geometry.polygon import rad, deg
from sympy.integrals.integrals import integrate
def test_eulerline():
    assert Triangle(Point(0, 0), Point(1, 0), Point(0, 1)).eulerline == Line(Point2D(0, 0), Point2D(S.Half, S.Half))
    assert Triangle(Point(0, 0), Point(10, 0), Point(5, 5 * sqrt(3))).eulerline == Point2D(5, 5 * sqrt(3) / 3)
    assert Triangle(Point(4, -6), Point(4, -1), Point(-3, 3)).eulerline == Line(Point2D(Rational(64, 7), 3), Point2D(Rational(-29, 14), Rational(-7, 2)))