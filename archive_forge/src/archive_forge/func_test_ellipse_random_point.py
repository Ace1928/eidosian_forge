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
def test_ellipse_random_point():
    y1 = Symbol('y1', real=True)
    e3 = Ellipse(Point(0, 0), y1, y1)
    rx, ry = (Symbol('rx'), Symbol('ry'))
    for ind in range(0, 5):
        r = e3.random_point()
        assert e3.equation(rx, ry).subs(zip((rx, ry), r.args)).equals(0)
    r = e3.random_point(seed=1)
    assert e3.equation(rx, ry).subs(zip((rx, ry), r.args)).equals(0)