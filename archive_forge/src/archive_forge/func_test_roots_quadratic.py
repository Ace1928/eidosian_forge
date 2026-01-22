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
def test_roots_quadratic():
    assert roots_quadratic(Poly(2 * x ** 2, x)) == [0, 0]
    assert roots_quadratic(Poly(2 * x ** 2 + 3 * x, x)) == [Rational(-3, 2), 0]
    assert roots_quadratic(Poly(2 * x ** 2 + 3, x)) == [-I * sqrt(6) / 2, I * sqrt(6) / 2]
    assert roots_quadratic(Poly(2 * x ** 2 + 4 * x + 3, x)) == [-1 - I * sqrt(2) / 2, -1 + I * sqrt(2) / 2]
    _check(Poly(2 * x ** 2 + 4 * x + 3, x).all_roots())
    f = x ** 2 + (2 * a * e + 2 * c * e) / (a - c) * x + (d - b + a * e ** 2 - c * e ** 2) / (a - c)
    assert roots_quadratic(Poly(f, x)) == [-e * (a + c) / (a - c) - sqrt(a * b + c * d - a * d - b * c + 4 * a * c * e ** 2) / (a - c), -e * (a + c) / (a - c) + sqrt(a * b + c * d - a * d - b * c + 4 * a * c * e ** 2) / (a - c)]
    f = Poly(y * x ** 2 - 2 * x - 2 * y, x)
    assert roots_quadratic(f) == [-sqrt(2 * y ** 2 + 1) / y + 1 / y, sqrt(2 * y ** 2 + 1) / y + 1 / y]
    f = Poly(x ** 2 + (-y ** 2 - 2) * x + y ** 2 + 1, x)
    assert roots_quadratic(f) == [1, y ** 2 + 1]
    f = Poly(sqrt(2) * x ** 2 - 1, x)
    r = roots_quadratic(f)
    assert r == _nsort(r)
    f = Poly(-24 * x ** 2 - 180 * x + 264)
    assert [w.n(2) for w in f.all_roots(radicals=True)] == [w.n(2) for w in f.all_roots(radicals=False)]
    for _a, _b, _c in product((-2, 2), (-2, 2), (0, -1)):
        f = Poly(_a * x ** 2 + _b * x + _c)
        roots = roots_quadratic(f)
        assert roots == _nsort(roots)