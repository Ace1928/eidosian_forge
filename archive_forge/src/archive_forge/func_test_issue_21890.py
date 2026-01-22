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
def test_issue_21890():
    e = S(2) / 3
    assert nonlinsolve([4 * x ** 3 * y ** 4 - 2 * y, 4 * x ** 4 * y ** 3 - 2 * x], x, y) == {(2 ** e / (2 * y), y), ((-2 ** e / 4 - 2 ** e * sqrt(3) * I / 4) / y, y), ((-2 ** e / 4 + 2 ** e * sqrt(3) * I / 4) / y, y)}
    assert nonlinsolve([(1 - 4 * x ** 2) * exp(-2 * x ** 2 - 2 * y ** 2), -4 * x * y * exp(-2 * x ** 2) * exp(-2 * y ** 2)], x, y) == {(-S(1) / 2, 0), (S(1) / 2, 0)}
    rx, ry = symbols('x y', real=True)
    sol = nonlinsolve([4 * rx ** 3 * ry ** 4 - 2 * ry, 4 * rx ** 4 * ry ** 3 - 2 * rx], rx, ry)
    ans = {(2 ** (S(2) / 3) / (2 * ry), ry), ((-2 ** (S(2) / 3) / 4 - 2 ** (S(2) / 3) * sqrt(3) * I / 4) / ry, ry), ((-2 ** (S(2) / 3) / 4 + 2 ** (S(2) / 3) * sqrt(3) * I / 4) / ry, ry)}
    assert sol == ans