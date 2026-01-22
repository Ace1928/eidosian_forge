from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Rational
from sympy.integrals.intpoly import (decompose, best_origin, distance_to_side,
from sympy.geometry.line import Segment2D
from sympy.geometry.polygon import Polygon
from sympy.geometry.point import Point, Point2D
from sympy.abc import x, y, z
from sympy.testing.pytest import slow
def test_point_sort():
    assert point_sort([Point(0, 0), Point(1, 0), Point(1, 1)]) == [Point2D(1, 1), Point2D(1, 0), Point2D(0, 0)]
    fig6 = Polygon((0, 0), (1, 0), (1, 1))
    assert polytope_integrate(fig6, x * y) == Rational(-1, 8)
    assert polytope_integrate(fig6, x * y, clockwise=True) == Rational(1, 8)