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
def test_exponential_real():
    from sympy.abc import y
    e1 = 3 ** (2 * x) - 2 ** (x + 3)
    e2 = 4 ** (5 - 9 * x) - 8 ** (2 - x)
    e3 = 2 ** x + 4 ** x
    e4 = exp(log(5) * x) - 2 ** x
    e5 = exp(x / y) * exp(-z / y) - 2
    e6 = 5 ** (x / 2) - 2 ** (x / 3)
    e7 = 4 ** (x + 1) + 4 ** (x + 2) + 4 ** (x - 1) - 3 ** (x + 2) - 3 ** (x + 3)
    e8 = -9 * exp(-2 * x + 5) + 4 * exp(3 * x + 1)
    e9 = 2 ** x + 4 ** x + 8 ** x - 84
    e10 = 29 * 2 ** (x + 1) * 615 ** x - 123 * 2726 ** x
    assert solveset(e1, x, S.Reals) == FiniteSet(-3 * log(2) / (-2 * log(3) + log(2)))
    assert solveset(e2, x, S.Reals) == FiniteSet(Rational(4, 15))
    assert solveset(e3, x, S.Reals) == S.EmptySet
    assert solveset(e4, x, S.Reals) == FiniteSet(0)
    assert solveset(e5, x, S.Reals) == Intersection(S.Reals, FiniteSet(y * log(2 * exp(z / y))))
    assert solveset(e6, x, S.Reals) == FiniteSet(0)
    assert solveset(e7, x, S.Reals) == FiniteSet(2)
    assert solveset(e8, x, S.Reals) == FiniteSet(-2 * log(2) / 5 + 2 * log(3) / 5 + Rational(4, 5))
    assert solveset(e9, x, S.Reals) == FiniteSet(2)
    assert solveset(e10, x, S.Reals) == FiniteSet((-log(29) - log(2) + log(123)) / (-log(2726) + log(2) + log(615)))
    assert solveset_real(-9 * exp(-2 * x + 5) + 2 ** (x + 1), x) == FiniteSet(-((-5 - 2 * log(3) + log(2)) / (log(2) + 2)))
    assert solveset_real(4 ** (x / 2) - 2 ** (x / 3), x) == FiniteSet(0)
    b = sqrt(6) * sqrt(log(2)) / sqrt(log(5))
    assert solveset_real(5 ** (x / 2) - 2 ** (3 / x), x) == FiniteSet(-b, b)
    C1, C2 = symbols('C1 C2')
    f = Function('f')
    assert solveset_real(C1 + C2 / x ** 2 - exp(-f(x)), f(x)) == Intersection(S.Reals, FiniteSet(-log(C1 + C2 / x ** 2)))
    y = symbols('y', positive=True)
    assert solveset_real(x ** 2 - y ** 2 / exp(x), y) == Intersection(S.Reals, FiniteSet(-sqrt(x ** 2 * exp(x)), sqrt(x ** 2 * exp(x))))
    p = Symbol('p', positive=True)
    assert solveset_real((1 / p + 1) ** (p + 1), p).dummy_eq(ConditionSet(x, Eq((1 + 1 / x) ** (x + 1), 0), S.Reals))
    assert solveset(2 ** x - 4 ** x + 12, x, S.Reals) == {2}
    assert solveset(2 ** x - 2 ** (2 * x) + 12, x, S.Reals) == {2}