from sympy.core import expand
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sec
from sympy.geometry.line import Segment2D
from sympy.geometry.point import Point2D
from sympy.geometry import (Circle, Ellipse, GeometryError, Line, Point,
from sympy.testing.pytest import raises, slow
from sympy.integrals.integrals import integrate
from sympy.functions.special.elliptic_integrals import elliptic_e
from sympy.functions.elementary.miscellaneous import Max
def test_is_tangent():
    e1 = Ellipse(Point(0, 0), 3, 5)
    c1 = Circle(Point(2, -2), 7)
    assert e1.is_tangent(Point(0, 0)) is False
    assert e1.is_tangent(Point(3, 0)) is False
    assert e1.is_tangent(e1) is True
    assert e1.is_tangent(Ellipse((0, 0), 1, 2)) is False
    assert e1.is_tangent(Ellipse((0, 0), 3, 2)) is True
    assert c1.is_tangent(Ellipse((2, -2), 7, 1)) is True
    assert c1.is_tangent(Circle((11, -2), 2)) is True
    assert c1.is_tangent(Circle((7, -2), 2)) is True
    assert c1.is_tangent(Ray((-5, -2), (-15, -20))) is False
    assert c1.is_tangent(Ray((-3, -2), (-15, -20))) is False
    assert c1.is_tangent(Ray((-3, -22), (15, 20))) is False
    assert c1.is_tangent(Ray((9, 20), (9, -20))) is True
    assert e1.is_tangent(Segment((2, 2), (-7, 7))) is False
    assert e1.is_tangent(Segment((0, 0), (1, 2))) is False
    assert c1.is_tangent(Segment((0, 0), (-5, -2))) is False
    assert e1.is_tangent(Segment((3, 0), (12, 12))) is False
    assert e1.is_tangent(Segment((12, 12), (3, 0))) is False
    assert e1.is_tangent(Segment((-3, 0), (3, 0))) is False
    assert e1.is_tangent(Segment((-3, 5), (3, 5))) is True
    assert e1.is_tangent(Line((10, 0), (10, 10))) is False
    assert e1.is_tangent(Line((0, 0), (1, 1))) is False
    assert e1.is_tangent(Line((-3, 0), (-2.99, -0.001))) is False
    assert e1.is_tangent(Line((-3, 0), (-3, 1))) is True
    assert e1.is_tangent(Polygon((0, 0), (5, 5), (5, -5))) is False
    assert e1.is_tangent(Polygon((-100, -50), (-40, -334), (-70, -52))) is False
    assert e1.is_tangent(Polygon((-3, 0), (3, 0), (0, 1))) is False
    assert e1.is_tangent(Polygon((-3, 0), (3, 0), (0, 5))) is False
    assert e1.is_tangent(Polygon((-3, 0), (0, -5), (3, 0), (0, 5))) is False
    assert e1.is_tangent(Polygon((-3, -5), (-3, 5), (3, 5), (3, -5))) is True
    assert c1.is_tangent(Polygon((-3, -5), (-3, 5), (3, 5), (3, -5))) is False
    assert e1.is_tangent(Polygon((0, 0), (3, 0), (7, 7), (0, 5))) is False
    assert e1.is_tangent(Polygon((3, 12), (3, -12), (6, 5))) is True
    assert e1.is_tangent(Polygon((3, 12), (3, -12), (0, -5), (0, 5))) is False
    assert e1.is_tangent(Polygon((3, 0), (5, 7), (6, -5))) is False
    raises(TypeError, lambda: e1.is_tangent(Point(0, 0, 0)))
    raises(TypeError, lambda: e1.is_tangent(Rational(5)))