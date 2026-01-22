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
def test_svg():
    a = Line(Point(1, 1), Point(1, 2))
    assert a._svg() == '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" stroke-width="2.0" opacity="0.6" d="M 1.00000000000000,1.00000000000000 L 1.00000000000000,2.00000000000000" marker-start="url(#markerReverseArrow)" marker-end="url(#markerArrow)"/>'
    a = Segment(Point(1, 0), Point(1, 1))
    assert a._svg() == '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" stroke-width="2.0" opacity="0.6" d="M 1.00000000000000,0 L 1.00000000000000,1.00000000000000" />'
    a = Ray(Point(2, 3), Point(3, 5))
    assert a._svg() == '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" stroke-width="2.0" opacity="0.6" d="M 2.00000000000000,3.00000000000000 L 3.00000000000000,5.00000000000000" marker-start="url(#markerCircle)" marker-end="url(#markerArrow)"/>'