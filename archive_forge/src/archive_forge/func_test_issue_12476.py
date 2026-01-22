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
def test_issue_12476():
    x0, x1, x2, x3, x4, x5 = symbols('x0 x1 x2 x3 x4 x5')
    eqns = [x0 ** 2 - x0, x0 * x1 - x1, x0 * x2 - x2, x0 * x3 - x3, x0 * x4 - x4, x0 * x5 - x5, x0 * x1 - x1, -x0 / 3 + x1 ** 2 - 2 * x2 / 3, x1 * x2 - x1 / 3 - x2 / 3 - x3 / 3, x1 * x3 - x2 / 3 - x3 / 3 - x4 / 3, x1 * x4 - 2 * x3 / 3 - x5 / 3, x1 * x5 - x4, x0 * x2 - x2, x1 * x2 - x1 / 3 - x2 / 3 - x3 / 3, -x0 / 6 - x1 / 6 + x2 ** 2 - x2 / 6 - x3 / 3 - x4 / 6, -x1 / 6 + x2 * x3 - x2 / 3 - x3 / 6 - x4 / 6 - x5 / 6, x2 * x4 - x2 / 3 - x3 / 3 - x4 / 3, x2 * x5 - x3, x0 * x3 - x3, x1 * x3 - x2 / 3 - x3 / 3 - x4 / 3, -x1 / 6 + x2 * x3 - x2 / 3 - x3 / 6 - x4 / 6 - x5 / 6, -x0 / 6 - x1 / 6 - x2 / 6 + x3 ** 2 - x3 / 3 - x4 / 6, -x1 / 3 - x2 / 3 + x3 * x4 - x3 / 3, -x2 + x3 * x5, x0 * x4 - x4, x1 * x4 - 2 * x3 / 3 - x5 / 3, x2 * x4 - x2 / 3 - x3 / 3 - x4 / 3, -x1 / 3 - x2 / 3 + x3 * x4 - x3 / 3, -x0 / 3 - 2 * x2 / 3 + x4 ** 2, -x1 + x4 * x5, x0 * x5 - x5, x1 * x5 - x4, x2 * x5 - x3, -x2 + x3 * x5, -x1 + x4 * x5, -x0 + x5 ** 2, x0 - 1]
    sols = [{x0: 1, x3: Rational(1, 6), x2: Rational(1, 6), x4: Rational(-2, 3), x1: Rational(-2, 3), x5: 1}, {x0: 1, x3: S.Half, x2: Rational(-1, 2), x4: 0, x1: 0, x5: -1}, {x0: 1, x3: Rational(-1, 3), x2: Rational(-1, 3), x4: Rational(1, 3), x1: Rational(1, 3), x5: 1}, {x0: 1, x3: 1, x2: 1, x4: 1, x1: 1, x5: 1}, {x0: 1, x3: Rational(-1, 3), x2: Rational(1, 3), x4: sqrt(5) / 3, x1: -sqrt(5) / 3, x5: -1}, {x0: 1, x3: Rational(-1, 3), x2: Rational(1, 3), x4: -sqrt(5) / 3, x1: sqrt(5) / 3, x5: -1}]
    assert solve(eqns) == sols