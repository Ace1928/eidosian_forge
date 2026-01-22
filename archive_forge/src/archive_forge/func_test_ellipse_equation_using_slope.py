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
def test_ellipse_equation_using_slope():
    from sympy.abc import x, y
    e1 = Ellipse(Point(1, 0), 3, 2)
    assert str(e1.equation(_slope=1)) == str((-x + y + 1) ** 2 / 8 + (x + y - 1) ** 2 / 18 - 1)
    e2 = Ellipse(Point(0, 0), 4, 1)
    assert str(e2.equation(_slope=1)) == str((-x + y) ** 2 / 2 + (x + y) ** 2 / 32 - 1)
    e3 = Ellipse(Point(1, 5), 6, 2)
    assert str(e3.equation(_slope=2)) == str((-2 * x + y - 3) ** 2 / 20 + (x + 2 * y - 11) ** 2 / 180 - 1)