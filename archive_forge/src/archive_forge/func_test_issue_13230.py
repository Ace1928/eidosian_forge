from sympy.core.function import expand
from sympy.core import (GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (AlgebraicNumber, I, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.solvers.solveset import nonlinsolve
from sympy.geometry import Circle, intersection
from sympy.testing.pytest import raises, slow
from sympy.sets.sets import FiniteSet
from sympy.geometry.point import Point2D
from sympy.polys.numberfields.minpoly import (
from sympy.polys.partfrac import apart
from sympy.polys.polyerrors import (
from sympy.polys.domains import QQ
from sympy.polys.rootoftools import rootof
from sympy.polys.polytools import degree
from sympy.abc import x, y, z
def test_issue_13230():
    c1 = Circle(Point2D(3, sqrt(5)), 5)
    c2 = Circle(Point2D(4, sqrt(7)), 6)
    assert intersection(c1, c2) == [Point2D(-1 + (-sqrt(7) + sqrt(5)) * (-2 * sqrt(7) / 29 + 9 * sqrt(5) / 29 + sqrt(196 * sqrt(35) + 1941) / 29), -2 * sqrt(7) / 29 + 9 * sqrt(5) / 29 + sqrt(196 * sqrt(35) + 1941) / 29), Point2D(-1 + (-sqrt(7) + sqrt(5)) * (-sqrt(196 * sqrt(35) + 1941) / 29 - 2 * sqrt(7) / 29 + 9 * sqrt(5) / 29), -sqrt(196 * sqrt(35) + 1941) / 29 - 2 * sqrt(7) / 29 + 9 * sqrt(5) / 29)]