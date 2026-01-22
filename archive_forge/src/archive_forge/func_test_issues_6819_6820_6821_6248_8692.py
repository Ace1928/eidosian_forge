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
def test_issues_6819_6820_6821_6248_8692():
    x, y = symbols('x y', real=True)
    assert solve(abs(x + 3) - 2 * abs(x - 3)) == [1, 9]
    assert solve([abs(x) - 2, arg(x) - pi], x) == [(-2,)]
    assert set(solve(abs(x - 7) - 8)) == {-S.One, S(15)}
    assert solve(Eq(Abs(x + 1) + Abs(x ** 2 - 7), 9), x) == [Rational(-1, 2) + sqrt(61) / 2, -sqrt(69) / 2 + S.Half]
    assert solve(2 * abs(x) - abs(x - 1)) == [-1, Rational(1, 3)]
    x = symbols('x')
    assert solve([re(x) - 1, im(x) - 2], x) == [{re(x): 1, x: 1 + 2 * I, im(x): 2}]
    eq = sqrt(re(x) ** 2 + im(x) ** 2) - 3
    assert solve(eq) == solve(eq, x)
    i = symbols('i', imaginary=True)
    assert solve(abs(i) - 3) == [-3 * I, 3 * I]
    raises(NotImplementedError, lambda: solve(abs(x) - 3))
    w = symbols('w', integer=True)
    assert solve(2 * x ** w - 4 * y ** w, w) == solve((x / y) ** w - 2, w)
    x, y = symbols('x y', real=True)
    assert solve(x + y * I + 3) == {y: 0, x: -3}
    assert solve(x * (1 + I)) == [0]
    x, y = symbols('x y', imaginary=True)
    assert solve(x + y * I + 3 + 2 * I) == {x: -2 * I, y: 3 * I}
    x = symbols('x', real=True)
    assert solve(x + y + 3 + 2 * I) == {x: -3, y: -2 * I}
    f = Function('f')
    assert solve(f(x + 1) - f(2 * x - 1)) == [2]
    assert solve(log(x + 1) - log(2 * x - 1)) == [2]
    x = symbols('x')
    assert solve(2 ** x + 4 ** x) == [I * pi / log(2)]