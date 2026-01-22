from sympy.core.containers import Tuple
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.hyperbolic import asinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry import Curve, Line, Point, Ellipse, Ray, Segment, Circle, Polygon, RegularPolygon
from sympy.testing.pytest import raises, slow
def test_issue_17997():
    t, s = symbols('t s')
    c = Curve((t, t ** 2), (t, 0, 10))
    p = Curve([2 * s, s ** 2], (s, 0, 2))
    assert c(2) == Point(2, 4)
    assert p(1) == Point(2, 1)