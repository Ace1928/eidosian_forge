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
def test_issue_20163():
    assert apart(1 / (x ** 6 + 1), extension=[sqrt(3), I]) == (sqrt(3) + I) / (2 * x + sqrt(3) + I) / 6 + (sqrt(3) - I) / (2 * x + sqrt(3) - I) / 6 - (sqrt(3) - I) / (2 * x - sqrt(3) + I) / 6 - (sqrt(3) + I) / (2 * x - sqrt(3) - I) / 6 + I / (x + I) / 6 - I / (x - I) / 6