from math import isclose
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda, nfloat, diff)
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, oo, pi, Integer)
from sympy.core.relational import (Eq, Gt, Ne, Ge)
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign, conjugate)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction,
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
from sympy.functions.special.error_functions import (erf, erfc,
from sympy.logic.boolalg import And
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.sets.contains import Contains
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import ImageSet, Range
from sympy.sets.sets import (Complement, FiniteSet,
from sympy.simplify import simplify
from sympy.tensor.indexed import Indexed
from sympy.utilities.iterables import numbered_symbols
from sympy.testing.pytest import (XFAIL, raises, skip, slow, SKIP, _both_exp_pow)
from sympy.core.random import verify_numerically as tn
from sympy.physics.units import cm
from sympy.solvers import solve
from sympy.solvers.solveset import (
from sympy.abc import (a, b, c, d, e, f, g, h, i, j, k, l, m, n, q, r,
def test_issue_18208():
    variables = symbols('x0:16') + symbols('y0:12')
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11 = variables
    eqs = [x0 + x1 + x2 + x3 - 51, x0 + x1 + x4 + x5 - 46, x2 + x3 + x6 + x7 - 39, x0 + x3 + x4 + x7 - 50, x1 + x2 + x5 + x6 - 35, x4 + x5 + x6 + x7 - 34, x4 + x5 + x8 + x9 - 46, x10 + x11 + x6 + x7 - 23, x11 + x4 + x7 + x8 - 25, x10 + x5 + x6 + x9 - 44, x10 + x11 + x8 + x9 - 35, x12 + x13 + x8 + x9 - 35, x10 + x11 + x14 + x15 - 29, x11 + x12 + x15 + x8 - 35, x10 + x13 + x14 + x9 - 29, x12 + x13 + x14 + x15 - 29, y0 + y1 + y2 + y3 - 55, y0 + y1 + y4 + y5 - 53, y2 + y3 + y6 + y7 - 56, y0 + y3 + y4 + y7 - 57, y1 + y2 + y5 + y6 - 52, y4 + y5 + y6 + y7 - 54, y4 + y5 + y8 + y9 - 48, y10 + y11 + y6 + y7 - 60, y11 + y4 + y7 + y8 - 51, y10 + y5 + y6 + y9 - 57, y10 + y11 + y8 + y9 - 54, x10 - 2, x11 - 5, x12 - 1, x13 - 6, x14 - 1, x15 - 21, y0 - 12, y1 - 20]
    expected = [38 - x3, x3 - 10, 23 - x3, x3, 12 - x7, x7 + 6, 16 - x7, x7, 8, 20, 2, 5, 1, 6, 1, 21, 12, 20, -y11 + y9 + 2, y11 - y9 + 21, -y11 - y7 + y9 + 24, y11 + y7 - y9 - 3, 33 - y7, y7, 27 - y9, y9, 27 - y11, y11]
    A, b = linear_eq_to_matrix(eqs, variables)
    solve_expected = {v: eq for v, eq in zip(variables, expected) if v != eq}
    assert solve(eqs, variables) == solve_expected
    linsolve_expected = FiniteSet(Tuple(*expected))
    assert linsolve(eqs, variables) == linsolve_expected
    assert linsolve((A, b), variables) == linsolve_expected
    gj_solve, new_vars = A.gauss_jordan_solve(b)
    gj_solve = list(gj_solve)
    gj_expected = linsolve_expected.subs(zip([x3, x7, y7, y9, y11], new_vars))
    assert FiniteSet(Tuple(*gj_solve)) == gj_expected
    nonlinsolve_expected = FiniteSet((38 - x3, x3 - 10, 23 - x3, x3, 12 - x7, x7 + 6, 16 - x7, x7, 8, 20, 2, 5, 1, 6, 1, 21, 12, 20, -y5 + y7 - 1, y5 - y7 + 24, 21 - y5, y5, 33 - y7, y7, 27 - y9, y9, -y5 + y7 - y9 + 24, y5 - y7 + y9 + 3))
    assert nonlinsolve(eqs, variables) == nonlinsolve_expected