from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.complexes import (conjugate, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.polys.domains.integerring import ZZ
from sympy.sets.sets import Interval
from sympy.simplify.powsimp import powsimp
from sympy.polys import Poly, cyclotomic_poly, intervals, nroots, rootof
from sympy.polys.polyroots import (root_factors, roots_linear,
from sympy.polys.orthopolys import legendre_poly
from sympy.polys.polyerrors import PolynomialError, \
from sympy.polys.polyutils import _nsort
from sympy.testing.pytest import raises, slow
from sympy.core.random import verify_numerically
import mpmath
from itertools import product
def test_roots_cubic():
    assert roots_cubic(Poly(2 * x ** 3, x)) == [0, 0, 0]
    assert roots_cubic(Poly(x ** 3 - 3 * x ** 2 + 3 * x - 1, x)) == [1, 1, 1]
    r = root(y, 3)
    assert roots_cubic(Poly(x ** 3 - y, x)) == [r, r * (-S.Half + sqrt(3) * I / 2), r * (-S.Half - sqrt(3) * I / 2)]
    assert roots_cubic(Poly(x ** 3 - -1, x)) == [-1, S.Half - I * sqrt(3) / 2, S.Half + I * sqrt(3) / 2]
    assert roots_cubic(Poly(2 * x ** 3 - 3 * x ** 2 - 3 * x - 1, x))[0] == S.Half + 3 ** Rational(1, 3) / 2 + 3 ** Rational(2, 3) / 2
    eq = -x ** 3 + 2 * x ** 2 + 3 * x - 2
    assert roots(eq, trig=True, multiple=True) == roots_cubic(Poly(eq, x), trig=True) == [Rational(2, 3) + 2 * sqrt(13) * cos(acos(8 * sqrt(13) / 169) / 3) / 3, -2 * sqrt(13) * sin(-acos(8 * sqrt(13) / 169) / 3 + pi / 6) / 3 + Rational(2, 3), -2 * sqrt(13) * cos(-acos(8 * sqrt(13) / 169) / 3 + pi / 3) / 3 + Rational(2, 3)]