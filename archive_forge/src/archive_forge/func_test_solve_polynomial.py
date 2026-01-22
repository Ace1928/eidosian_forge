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
def test_solve_polynomial():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert solveset_real(3 * x - 2, x) == FiniteSet(Rational(2, 3))
    assert solveset_real(x ** 2 - 1, x) == FiniteSet(-S.One, S.One)
    assert solveset_real(x - y ** 3, x) == FiniteSet(y ** 3)
    assert solveset_real(x ** 3 - 15 * x - 4, x) == FiniteSet(-2 + 3 ** S.Half, S(4), -2 - 3 ** S.Half)
    assert solveset_real(sqrt(x) - 1, x) == FiniteSet(1)
    assert solveset_real(sqrt(x) - 2, x) == FiniteSet(4)
    assert solveset_real(x ** Rational(1, 4) - 2, x) == FiniteSet(16)
    assert solveset_real(x ** Rational(1, 3) - 3, x) == FiniteSet(27)
    assert len(solveset_real(x ** 5 + x ** 3 + 1, x)) == 1
    assert len(solveset_real(-2 * x ** 3 + 4 * x ** 2 - 2 * x + 6, x)) > 0
    assert solveset_real(x ** 6 + x ** 4 + I, x) is S.EmptySet