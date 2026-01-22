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
def test_line_intersection():
    x0 = tan(pi * Rational(13, 45))
    x1 = sqrt(3)
    x2 = x0 ** 2
    x, y = [8 * x0 / (x0 + x1), (24 * x0 - 8 * x1 * x2) / (x2 - 3)]
    assert Line(Point(0, 0), Point(1, -sqrt(3))).contains(Point(x, y)) is True