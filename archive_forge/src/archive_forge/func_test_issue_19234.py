from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Rational
from sympy.integrals.intpoly import (decompose, best_origin, distance_to_side,
from sympy.geometry.line import Segment2D
from sympy.geometry.polygon import Polygon
from sympy.geometry.point import Point, Point2D
from sympy.abc import x, y, z
from sympy.testing.pytest import slow
def test_issue_19234():
    polygon = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
    polys = [1, x, y, x * y, x ** 2 * y, x * y ** 2]
    assert polytope_integrate(polygon, polys) == {1: 1, x: S.Half, y: S.Half, x * y: Rational(1, 4), x ** 2 * y: Rational(1, 6), x * y ** 2: Rational(1, 6)}
    polys = [1, x, y, x * y, 3 + x ** 2 * y, x + x * y ** 2]
    assert polytope_integrate(polygon, polys) == {1: 1, x: S.Half, y: S.Half, x * y: Rational(1, 4), x ** 2 * y + 3: Rational(19, 6), x * y ** 2 + x: Rational(2, 3)}