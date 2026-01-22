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
def test_object_from_equation():
    from sympy.abc import x, y, a, b
    assert Line(3 * x + y + 18) == Line2D(Point2D(0, -18), Point2D(1, -21))
    assert Line(3 * x + 5 * y + 1) == Line2D(Point2D(0, Rational(-1, 5)), Point2D(1, Rational(-4, 5)))
    assert Line(3 * a + b + 18, x='a', y='b') == Line2D(Point2D(0, -18), Point2D(1, -21))
    assert Line(3 * x + y) == Line2D(Point2D(0, 0), Point2D(1, -3))
    assert Line(x + y) == Line2D(Point2D(0, 0), Point2D(1, -1))
    assert Line(Eq(3 * a + b, -18), x='a', y=b) == Line2D(Point2D(0, -18), Point2D(1, -21))
    assert Line(x - 1) == Line2D(Point2D(1, 0), Point2D(1, 1))
    assert Line(2 * x - 2, y=x) == Line2D(Point2D(0, 1), Point2D(1, 1))
    assert Line(y) == Line2D(Point2D(0, 0), Point2D(1, 0))
    assert Line(2 * y, x=y) == Line2D(Point2D(0, 0), Point2D(0, 1))
    assert Line(y, x=y) == Line2D(Point2D(0, 0), Point2D(0, 1))
    raises(ValueError, lambda: Line(x / y))
    raises(ValueError, lambda: Line(a / b, x='a', y='b'))
    raises(ValueError, lambda: Line(y / x))
    raises(ValueError, lambda: Line(b / a, x='a', y='b'))
    raises(ValueError, lambda: Line((x + 1) ** 2 + y))