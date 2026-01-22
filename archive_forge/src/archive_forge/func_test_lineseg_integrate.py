from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Rational
from sympy.integrals.intpoly import (decompose, best_origin, distance_to_side,
from sympy.geometry.line import Segment2D
from sympy.geometry.polygon import Polygon
from sympy.geometry.point import Point, Point2D
from sympy.abc import x, y, z
from sympy.testing.pytest import slow
def test_lineseg_integrate():
    polygon = [(0, 5, 0), (5, 5, 0), (5, 5, 5), (0, 5, 5)]
    line_seg = [(0, 5, 0), (5, 5, 0)]
    assert lineseg_integrate(polygon, 0, line_seg, 1, 0) == 5
    assert lineseg_integrate(polygon, 0, line_seg, 0, 0) == 0