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
def test_bisectors():
    p1, p2, p3 = (Point(0, 0), Point(1, 0), Point(0, 1))
    p = Polygon(Point(0, 0), Point(2, 0), Point(1, 1), Point(0, 3))
    q = Polygon(Point(1, 0), Point(2, 0), Point(3, 3), Point(-1, 5))
    poly = Polygon(Point(3, 4), Point(0, 0), Point(8, 7), Point(-1, 1), Point(19, -19))
    t = Triangle(p1, p2, p3)
    assert t.bisectors()[p2] == Segment(Point(1, 0), Point(0, sqrt(2) - 1))
    assert p.bisectors()[Point2D(0, 3)] == Ray2D(Point2D(0, 3), Point2D(sin(acos(2 * sqrt(5) / 5) / 2), 3 - cos(acos(2 * sqrt(5) / 5) / 2)))
    assert q.bisectors()[Point2D(-1, 5)] == Ray2D(Point2D(-1, 5), Point2D(-1 + sqrt(29) * (5 * sin(acos(9 * sqrt(145) / 145) / 2) + 2 * cos(acos(9 * sqrt(145) / 145) / 2)) / 29, sqrt(29) * (-5 * cos(acos(9 * sqrt(145) / 145) / 2) + 2 * sin(acos(9 * sqrt(145) / 145) / 2)) / 29 + 5))
    assert poly.bisectors()[Point2D(-1, 1)] == Ray2D(Point2D(-1, 1), Point2D(-1 + sin(acos(sqrt(26) / 26) / 2 + pi / 4), 1 - sin(-acos(sqrt(26) / 26) / 2 + pi / 4)))