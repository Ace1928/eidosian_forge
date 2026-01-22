from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Rational
from sympy.integrals.intpoly import (decompose, best_origin, distance_to_side,
from sympy.geometry.line import Segment2D
from sympy.geometry.polygon import Polygon
from sympy.geometry.point import Point, Point2D
from sympy.abc import x, y, z
from sympy.testing.pytest import slow
def test_is_vertex():
    assert is_vertex(2) is False
    assert is_vertex((2, 3)) is True
    assert is_vertex(Point(2, 3)) is True
    assert is_vertex((2, 3, 4)) is True
    assert is_vertex((2, 3, 4, 5)) is False