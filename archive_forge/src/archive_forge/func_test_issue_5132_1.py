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
def test_issue_5132_1():
    system = [sqrt(x ** 2 + y ** 2) - sqrt(10), x + y - 4]
    assert nonlinsolve(system, [x, y]) == FiniteSet((1, 3), (3, 1))
    n = Dummy('n')
    eqs = [exp(x) ** 2 - sin(y) + z ** 2, 1 / exp(y) - 3]
    s_real_y = -log(3)
    s_real_z = sqrt(-exp(2 * x) - sin(log(3)))
    soln_real = FiniteSet((s_real_y, s_real_z), (s_real_y, -s_real_z))
    lam = Lambda(n, 2 * n * I * pi + -log(3))
    s_complex_y = ImageSet(lam, S.Integers)
    lam = Lambda(n, sqrt(-exp(2 * x) + sin(2 * n * I * pi + -log(3))))
    s_complex_z_1 = ImageSet(lam, S.Integers)
    lam = Lambda(n, -sqrt(-exp(2 * x) + sin(2 * n * I * pi + -log(3))))
    s_complex_z_2 = ImageSet(lam, S.Integers)
    soln_complex = FiniteSet((s_complex_y, s_complex_z_1), (s_complex_y, s_complex_z_2))
    soln = soln_real + soln_complex
    assert dumeq(nonlinsolve(eqs, [y, z]), soln)