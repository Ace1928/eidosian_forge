from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Rational
from sympy.integrals.intpoly import (decompose, best_origin, distance_to_side,
from sympy.geometry.line import Segment2D
from sympy.geometry.polygon import Polygon
from sympy.geometry.point import Point, Point2D
from sympy.abc import x, y, z
from sympy.testing.pytest import slow
def test_integration_reduction_dynamic():
    triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    facets = triangle.sides
    a, b = hyperplane_parameters(triangle)[0]
    x0 = facets[0].points[0]
    monomial_values = [[0, 0, 0, 0], [1, 0, 0, 5], [y, 0, 1, 15], [x, 1, 0, None]]
    assert integration_reduction_dynamic(facets, 0, a, b, x, 1, (x, y), 1, 0, 1, x0, monomial_values, 3) == Rational(25, 2)
    assert integration_reduction_dynamic(facets, 0, a, b, 0, 1, (x, y), 1, 0, 1, x0, monomial_values, 3) == 0