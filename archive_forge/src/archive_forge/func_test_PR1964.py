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
def test_PR1964():
    assert solve(sqrt(x)) == solve(sqrt(x ** 3)) == [0]
    assert solve(sqrt(x - 1)) == [1]
    a = Symbol('a')
    assert solve(-3 * a / sqrt(x), x) == []
    assert solve(2 * x / (x + 2) - 1, x) == [2]
    assert set(solve((x ** 2 / (7 - x)).diff(x))) == {S.Zero, S(14)}
    f = Function('f')
    assert solve((3 - 5 * x / f(x)) * f(x), f(x)) == [x * Rational(5, 3)]
    assert solve(1 / root(5 + x, 5) - 9, x) == [Rational(-295244, 59049)]
    assert solve(sqrt(x) + sqrt(sqrt(x)) - 4) == [(Rational(-1, 2) + sqrt(17) / 2) ** 4]
    assert set(solve(Poly(sqrt(exp(x)) + sqrt(exp(-x)) - 4))) in [{log((-sqrt(3) + 2) ** 2), log((sqrt(3) + 2) ** 2)}, {2 * log(-sqrt(3) + 2), 2 * log(sqrt(3) + 2)}, {log(-4 * sqrt(3) + 7), log(4 * sqrt(3) + 7)}]
    assert set(solve(Poly(exp(x) + exp(-x) - 4))) == {log(-sqrt(3) + 2), log(sqrt(3) + 2)}
    assert set(solve(x ** y + x ** (2 * y) - 1, x)) == {(Rational(-1, 2) + sqrt(5) / 2) ** (1 / y), (Rational(-1, 2) - sqrt(5) / 2) ** (1 / y)}
    assert solve(exp(x / y) * exp(-z / y) - 2, y) == [(x - z) / log(2)]
    assert solve(x ** z * y ** z - 2, z) in [[log(2) / (log(x) + log(y))], [log(2) / log(x * y)]]
    E = S.Exp1
    assert solve(exp(3 * x) - exp(3), x) in [[1, log(E * (Rational(-1, 2) - sqrt(3) * I / 2)), log(E * (Rational(-1, 2) + sqrt(3) * I / 2))], [1, log(-E / 2 - sqrt(3) * E * I / 2), log(-E / 2 + sqrt(3) * E * I / 2)]]
    p = Symbol('p', positive=True)
    assert solve((1 / p + 1) ** (p + 1)) == []