from sympy.core.numbers import (Float, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.sets import EmptySet
from sympy.simplify.simplify import simplify
from sympy.functions.elementary.trigonometric import tan
from sympy.geometry import (Circle, GeometryError, Line, Point, Ray,
from sympy.geometry.line import Undecidable
from sympy.geometry.polygon import _asa as asa
from sympy.utilities.iterables import cartes
from sympy.testing.pytest import raises, warns
def test_distance_2d():
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    half = S.Half
    s1 = Segment(Point(0, 0), Point(1, 1))
    s2 = Segment(Point(half, half), Point(1, 0))
    r = Ray(p1, p2)
    assert s1.distance(Point(0, 0)) == 0
    assert s1.distance((0, 0)) == 0
    assert s2.distance(Point(0, 0)) == 2 ** half / 2
    assert s2.distance(Point(Rational(3) / 2, Rational(3) / 2)) == 2 ** half
    assert Line(p1, p2).distance(Point(-1, 1)) == sqrt(2)
    assert Line(p1, p2).distance(Point(1, -1)) == sqrt(2)
    assert Line(p1, p2).distance(Point(2, 2)) == 0
    assert Line(p1, p2).distance((-1, 1)) == sqrt(2)
    assert Line((0, 0), (0, 1)).distance(p1) == 0
    assert Line((0, 0), (0, 1)).distance(p2) == 1
    assert Line((0, 0), (1, 0)).distance(p1) == 0
    assert Line((0, 0), (1, 0)).distance(p2) == 1
    assert r.distance(Point(-1, -1)) == sqrt(2)
    assert r.distance(Point(1, 1)) == 0
    assert r.distance(Point(-1, 1)) == sqrt(2)
    assert Ray((1, 1), (2, 2)).distance(Point(1.5, 3)) == 3 * sqrt(2) / 4
    assert r.distance((1, 1)) == 0