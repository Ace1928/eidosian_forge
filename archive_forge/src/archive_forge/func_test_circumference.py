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
def test_circumference():
    M = Symbol('M')
    m = Symbol('m')
    assert Ellipse(Point(0, 0), M, m).circumference == 4 * M * elliptic_e((M ** 2 - m ** 2) / M ** 2)
    assert Ellipse(Point(0, 0), 5, 4).circumference == 20 * elliptic_e(S(9) / 25)
    assert Ellipse(None, 1, None, 0).circumference == 2 * pi
    assert abs(Ellipse(None, hradius=5, vradius=3).circumference.evalf(16) - 25.52699886339813) < 1e-10