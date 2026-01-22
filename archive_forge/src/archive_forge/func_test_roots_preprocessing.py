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
def test_roots_preprocessing():
    f = a * y * x ** 2 + y - b
    coeff, poly = preprocess_roots(Poly(f, x))
    assert coeff == 1
    assert poly == Poly(a * y * x ** 2 + y - b, x)
    f = c ** 3 * x ** 3 + c ** 2 * x ** 2 + c * x + a
    coeff, poly = preprocess_roots(Poly(f, x))
    assert coeff == 1 / c
    assert poly == Poly(x ** 3 + x ** 2 + x + a, x)
    f = c ** 3 * x ** 3 + c ** 2 * x ** 2 + a
    coeff, poly = preprocess_roots(Poly(f, x))
    assert coeff == 1 / c
    assert poly == Poly(x ** 3 + x ** 2 + a, x)
    f = c ** 3 * x ** 3 + c * x + a
    coeff, poly = preprocess_roots(Poly(f, x))
    assert coeff == 1 / c
    assert poly == Poly(x ** 3 + x + a, x)
    f = c ** 3 * x ** 3 + a
    coeff, poly = preprocess_roots(Poly(f, x))
    assert coeff == 1 / c
    assert poly == Poly(x ** 3 + a, x)
    E, F, J, L = symbols('E,F,J,L')
    f = -21601054687500000000 * E ** 8 * J ** 8 / L ** 16 + 508232812500000000 * F * x * E ** 7 * J ** 7 / L ** 14 - 4269543750000000 * E ** 6 * F ** 2 * J ** 6 * x ** 2 / L ** 12 + 16194716250000 * E ** 5 * F ** 3 * J ** 5 * x ** 3 / L ** 10 - 27633173750 * E ** 4 * F ** 4 * J ** 4 * x ** 4 / L ** 8 + 14840215 * E ** 3 * F ** 5 * J ** 3 * x ** 5 / L ** 6 + 54794 * E ** 2 * F ** 6 * J ** 2 * x ** 6 / (5 * L ** 4) - 1153 * E * J * F ** 7 * x ** 7 / (80 * L ** 2) + 633 * F ** 8 * x ** 8 / 160000
    coeff, poly = preprocess_roots(Poly(f, x))
    assert coeff == 20 * E * J / (F * L ** 2)
    assert poly == 633 * x ** 8 - 115300 * x ** 7 + 4383520 * x ** 6 + 296804300 * x ** 5 - 27633173750 * x ** 4 + 809735812500 * x ** 3 - 10673859375000 * x ** 2 + 63529101562500 * x - 135006591796875
    f = Poly(-y ** 2 + x ** 2 * exp(x), y, domain=ZZ[x, exp(x)])
    g = Poly(-y ** 2 + exp(x), y, domain=ZZ[exp(x)])
    assert preprocess_roots(f) == (x, g)