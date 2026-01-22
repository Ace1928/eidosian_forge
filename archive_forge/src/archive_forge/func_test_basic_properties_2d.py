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
def test_basic_properties_2d():
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    p10 = Point(2000, 2000)
    p_r3 = Ray(p1, p2).random_point()
    p_r4 = Ray(p2, p1).random_point()
    l1 = Line(p1, p2)
    l3 = Line(Point(x1, x1), Point(x1, 1 + x1))
    l4 = Line(p1, Point(1, 0))
    r1 = Ray(p1, Point(0, 1))
    r2 = Ray(Point(0, 1), p1)
    s1 = Segment(p1, p10)
    p_s1 = s1.random_point()
    assert Line((1, 1), slope=1) == Line((1, 1), (2, 2))
    assert Line((1, 1), slope=oo) == Line((1, 1), (1, 2))
    assert Line((1, 1), slope=oo).bounds == (1, 1, 1, 2)
    assert Line((1, 1), slope=-oo) == Line((1, 1), (1, 2))
    assert Line(p1, p2).scale(2, 1) == Line(p1, Point(2, 1))
    assert Line(p1, p2) == Line(p1, p2)
    assert Line(p1, p2) != Line(p2, p1)
    assert l1 != Line(Point(x1, x1), Point(y1, y1))
    assert l1 != l3
    assert Line(p1, p10) != Line(p10, p1)
    assert Line(p1, p10) != p1
    assert p1 in l1
    assert p1 not in l3
    assert s1 in Line(p1, p10)
    assert Ray(Point(0, 0), Point(0, 1)) in Ray(Point(0, 0), Point(0, 2))
    assert Ray(Point(0, 0), Point(0, 2)) in Ray(Point(0, 0), Point(0, 1))
    assert Ray(Point(0, 0), Point(0, 2)).xdirection == S.Zero
    assert Ray(Point(0, 0), Point(1, 2)).xdirection == S.Infinity
    assert Ray(Point(0, 0), Point(-1, 2)).xdirection == S.NegativeInfinity
    assert Ray(Point(0, 0), Point(2, 0)).ydirection == S.Zero
    assert Ray(Point(0, 0), Point(2, 2)).ydirection == S.Infinity
    assert Ray(Point(0, 0), Point(2, -2)).ydirection == S.NegativeInfinity
    assert (r1 in s1) is False
    assert Segment(p1, p2) in s1
    assert Ray(Point(x1, x1), Point(x1, 1 + x1)) != Ray(p1, Point(-1, 5))
    assert Segment(p1, p2).midpoint == Point(S.Half, S.Half)
    assert Segment(p1, Point(-x1, x1)).length == sqrt(2 * x1 ** 2)
    assert l1.slope == 1
    assert l3.slope is oo
    assert l4.slope == 0
    assert Line(p1, Point(0, 1)).slope is oo
    assert Line(r1.source, r1.random_point()).slope == r1.slope
    assert Line(r2.source, r2.random_point()).slope == r2.slope
    assert Segment(Point(0, -1), Segment(p1, Point(0, 1)).random_point()).slope == Segment(p1, Point(0, 1)).slope
    assert l4.coefficients == (0, 1, 0)
    assert Line((-x, x), (-x + 1, x - 1)).coefficients == (1, 1, 0)
    assert Line(p1, Point(0, 1)).coefficients == (1, 0, 0)
    r = Ray((0, 0), angle=x)
    assert r.subs(x, 3 * pi / 4) == Ray((0, 0), (-1, 1))
    assert r.subs(x, 5 * pi / 4) == Ray((0, 0), (-1, -1))
    assert r.subs(x, -pi / 4) == Ray((0, 0), (1, -1))
    assert r.subs(x, pi / 2) == Ray((0, 0), (0, 1))
    assert r.subs(x, -pi / 2) == Ray((0, 0), (0, -1))
    for ind in range(0, 5):
        assert l3.random_point() in l3
    assert p_r3.x >= p1.x and p_r3.y >= p1.y
    assert p_r4.x <= p2.x and p_r4.y <= p2.y
    assert p1.x <= p_s1.x <= p10.x and p1.y <= p_s1.y <= p10.y
    assert hash(s1) != hash(Segment(p10, p1))
    assert s1.plot_interval() == [t, 0, 1]
    assert Line(p1, p10).plot_interval() == [t, -5, 5]
    assert Ray((0, 0), angle=pi / 4).plot_interval() == [t, 0, 10]