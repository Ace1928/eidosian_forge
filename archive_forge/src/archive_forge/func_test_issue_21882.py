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
def test_issue_21882():
    a, b, c, d, f, g, k = unknowns = symbols('a, b, c, d, f, g, k')
    equations = [-k * a + b + 5 * f / 6 + 2 * c / 9 + 5 * d / 6 + 4 * a / 3, -k * f + 4 * f / 3 + d / 2, -k * d + f / 6 + d, 13 * b / 18 + 13 * c / 18 + 13 * a / 18, -k * c + b / 2 + 20 * c / 9 + a, -k * b + b + c / 18 + a / 6, 5 * b / 3 + c / 3 + a, 2 * b / 3 + 2 * c + 4 * a / 3, -g]
    answer = [{a: 0, f: 0, b: 0, d: 0, c: 0, g: 0}, {a: 0, f: -d, b: 0, k: S(5) / 6, c: 0, g: 0}, {a: -2 * c, f: 0, b: c, d: 0, k: S(13) / 18, g: 0}]
    got = solve(equations, unknowns, dict=True)
    assert got == answer, (got, answer)