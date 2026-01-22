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
def test_do_poly_distance():
    square1 = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
    triangle1 = Polygon(Point(1, 2), Point(2, 2), Point(2, 1))
    assert square1._do_poly_distance(triangle1) == sqrt(2) / 2
    square2 = Polygon(Point(1, 0), Point(2, 0), Point(2, 1), Point(1, 1))
    with warns(UserWarning, match='Polygons may intersect producing erroneous output', test_stacklevel=False):
        assert square1._do_poly_distance(square2) == 0
    triangle2 = Polygon(Point(0, -1), Point(2, -1), Point(S.Half, S.Half))
    with warns(UserWarning, match='Polygons may intersect producing erroneous output', test_stacklevel=False):
        assert triangle2._do_poly_distance(square1) == 0