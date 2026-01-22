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
def test_arbitrary_point():
    l1 = Line3D(Point3D(0, 0, 0), Point3D(1, 1, 1))
    l2 = Line(Point(x1, x1), Point(y1, y1))
    assert l2.arbitrary_point() in l2
    assert Ray((1, 1), angle=pi / 4).arbitrary_point() == Point(t + 1, t + 1)
    assert Segment((1, 1), (2, 3)).arbitrary_point() == Point(1 + t, 1 + 2 * t)
    assert l1.perpendicular_segment(l1.arbitrary_point()) == l1.arbitrary_point()
    assert Ray3D((1, 1, 1), direction_ratio=[1, 2, 3]).arbitrary_point() == Point3D(t + 1, 2 * t + 1, 3 * t + 1)
    assert Segment3D(Point3D(0, 0, 0), Point3D(1, 1, 1)).midpoint == Point3D(S.Half, S.Half, S.Half)
    assert Segment3D(Point3D(x1, x1, x1), Point3D(y1, y1, y1)).length == sqrt(3) * sqrt((x1 - y1) ** 2)
    assert Segment3D((1, 1, 1), (2, 3, 4)).arbitrary_point() == Point3D(t + 1, 2 * t + 1, 3 * t + 1)
    raises(ValueError, lambda: Line((x, 1), (2, 3)).arbitrary_point(x))