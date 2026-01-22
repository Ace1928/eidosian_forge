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
def test_issue_15797_equals():
    Ri = 0.024127189424130748
    Ci = (0.0864931002830291, 0.0819863295239654)
    A = Point(0, 0.0578591400998346)
    c = Circle(Ci, Ri)
    assert c.is_tangent(c.tangent_lines(A)[0]) == True
    assert c.center.x.is_Rational
    assert c.center.y.is_Rational
    assert c.radius.is_Rational
    u = Circle(Ci, Ri, evaluate=False)
    assert u.center.x.is_Float
    assert u.center.y.is_Float
    assert u.radius.is_Float