from sympy.assumptions.ask import (Q, ask)
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core import (GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (E, Float, I, Rational, oo, pi)
from sympy.core.relational import (Eq, Gt, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (atanh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (cbrt, root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, atan2, cos, sec, sin, tan)
from sympy.functions.special.error_functions import (erf, erfc, erfcinv, erfinv)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import Matrix
from sympy.matrices import SparseMatrix
from sympy.polys.polytools import Poly
from sympy.printing.str import sstr
from sympy.simplify.radsimp import denom
from sympy.solvers.solvers import (nsolve, solve, solve_linear)
from sympy.core.function import nfloat
from sympy.solvers import solve_linear_system, solve_linear_system_LU, \
from sympy.solvers.bivariate import _filtered_gens, _solve_lambert, _lambert
from sympy.solvers.solvers import _invert, unrad, checksol, posify, _ispow, \
from sympy.physics.units import cm
from sympy.polys.rootoftools import CRootOf
from sympy.testing.pytest import slow, XFAIL, SKIP, raises
from sympy.core.random import verify_numerically as tn
from sympy.abc import a, b, c, d, e, k, h, p, x, y, z, t, q, m, R
@slow
def test_lambert_bivariate():
    assert solve((x ** 2 + x) * exp(x ** 2 + x) - 1) == [Rational(-1, 2) + sqrt(1 + 4 * LambertW(1)) / 2, Rational(-1, 2) - sqrt(1 + 4 * LambertW(1)) / 2]
    assert solve((x ** 2 + x) * exp((x ** 2 + x) * 2) - 1) == [Rational(-1, 2) + sqrt(1 + 2 * LambertW(2)) / 2, Rational(-1, 2) - sqrt(1 + 2 * LambertW(2)) / 2]
    assert solve(a / x + exp(x / 2), x) == [2 * LambertW(-a / 2)]
    assert solve((a / x + exp(x / 2)).diff(x), x) == [4 * LambertW(-sqrt(2) * sqrt(a) / 4), 4 * LambertW(sqrt(2) * sqrt(a) / 4)]
    assert solve((1 / x + exp(x / 2)).diff(x), x) == [4 * LambertW(-sqrt(2) / 4), 4 * LambertW(sqrt(2) / 4), 4 * LambertW(-sqrt(2) / 4, -1)]
    assert solve(x * log(x) + 3 * x + 1, x) == [exp(-3 + LambertW(-exp(3)))]
    assert solve(-x ** 2 + 2 ** x, x) == [2, 4, -2 * LambertW(log(2) / 2) / log(2)]
    assert solve(x ** 2 - 2 ** x, x) == [2, 4, -2 * LambertW(log(2) / 2) / log(2)]
    ans = solve(3 * x + 5 + 2 ** (-5 * x + 3), x)
    assert len(ans) == 1 and ans[0].expand() == Rational(-5, 3) + LambertW(-10240 * root(2, 3) * log(2) / 3) / (5 * log(2))
    assert solve(5 * x - 1 + 3 * exp(2 - 7 * x), x) == [Rational(1, 5) + LambertW(-21 * exp(Rational(3, 5)) / 5) / 7]
    assert solve((log(x) + x).subs(x, x ** 2 + 1)) == [-I * sqrt(-LambertW(1) + 1), sqrt(-1 + LambertW(1))]
    ax = a ** (3 * x + 5)
    ans = solve(3 * log(ax) + b * log(ax) + ax, x)
    x0 = 1 / log(a)
    x1 = sqrt(3) * I
    x2 = b + 3
    x3 = x2 * LambertW(1 / x2) / a ** 5
    x4 = x3 ** Rational(1, 3) / 2
    assert ans == [x0 * log(x4 * (-x1 - 1)), x0 * log(x4 * (x1 - 1)), x0 * log(x3) / 3]
    x1 = LambertW(Rational(1, 3))
    x2 = a ** (-5)
    x3 = -3 ** Rational(1, 3)
    x4 = 3 ** Rational(5, 6) * I
    x5 = x1 ** Rational(1, 3) * x2 ** Rational(1, 3) / 2
    ans = solve(3 * log(ax) + ax, x)
    assert ans == [x0 * log(3 * x1 * x2) / 3, x0 * log(x5 * (x3 - x4)), x0 * log(x5 * (x3 + x4))]
    p = symbols('p', positive=True)
    eq = 4 * 2 ** (2 * p + 3) - 2 * p - 3
    assert _solve_lambert(eq, p, _filtered_gens(Poly(eq), p)) == [Rational(-3, 2) - LambertW(-4 * log(2)) / (2 * log(2))]
    assert set(solve(3 ** cos(x) - cos(x) ** 3)) == {acos(3), acos(-3 * LambertW(-log(3) / 3) / log(3))}
    assert solve(2 * log(x) - 2 * log(z) + log(z + log(x) + log(z)), x) == [exp(-z + LambertW(2 * z ** 4 * exp(2 * z)) / 2) / z]
    ans = solve((a / x + exp(x / 2)).diff(x, 2), x)
    x0 = (-a) ** Rational(1, 3)
    x1 = sqrt(3) * I
    x2 = x0 / 6
    assert ans == [6 * LambertW(x0 / 3), 6 * LambertW(x2 * (-x1 - 1)), 6 * LambertW(x2 * (x1 - 1))]
    assert solve((1 / x + exp(x / 2)).diff(x, 2), x) == [6 * LambertW(Rational(-1, 3)), 6 * LambertW(Rational(1, 6) - sqrt(3) * I / 6), 6 * LambertW(Rational(1, 6) + sqrt(3) * I / 6), 6 * LambertW(Rational(-1, 3), -1)]
    assert solve(x ** 2 - y ** 2 / exp(x), x, y, dict=True) == [{x: 2 * LambertW(-y / 2)}, {x: 2 * LambertW(y / 2)}]
    assert solve((x ** 3) ** (x / 2) + pi / 2, x) == [exp(LambertW(-2 * log(2) / 3 + 2 * log(pi) / 3 + I * pi * Rational(2, 3)))]
    assert solve(1 / log(sqrt(x) + 2) ** 2 - 1 / x) == [(LambertW(-exp(-2), -1) + 2) ** 2]
    assert solve(1 / log(1 / sqrt(x) + 2) ** 2 - x) == [(LambertW(-exp(-2), -1) + 2) ** (-2)]
    assert solve(1 / log(x ** 2 + 2) ** 2 - x ** (-4)) == [-I * sqrt(2 - LambertW(exp(2))), -I * sqrt(LambertW(-exp(-2)) + 2), sqrt(-2 - LambertW(-exp(-2))), sqrt(-2 + LambertW(exp(2))), -sqrt(-2 - LambertW(-exp(-2), -1)), sqrt(-2 - LambertW(-exp(-2), -1))]