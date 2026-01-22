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
@_both_exp_pow
def test_multi_exp():
    k1, k2, k3 = symbols('k1, k2, k3')
    assert dumeq(solveset(exp(exp(x)) - 5, x), imageset(Lambda(((k1, n),), I * (2 * k1 * pi + arg(2 * n * I * pi + log(5))) + log(Abs(2 * n * I * pi + log(5)))), ProductSet(S.Integers, S.Integers)))
    assert dumeq(solveset(d * exp(exp(a * x + b)) + c, x), imageset(Lambda(x, (-b + x) / a), ImageSet(Lambda(((k1, n),), I * (2 * k1 * pi + arg(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d)))) + log(Abs(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d))))), ProductSet(S.Integers, S.Integers))))
    assert dumeq(solveset(d * exp(exp(exp(a * x + b))) + c, x), imageset(Lambda(x, (-b + x) / a), ImageSet(Lambda(((k2, k1, n),), I * (2 * k2 * pi + arg(I * (2 * k1 * pi + arg(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d)))) + log(Abs(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d)))))) + log(Abs(I * (2 * k1 * pi + arg(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d)))) + log(Abs(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d))))))), ProductSet(S.Integers, S.Integers, S.Integers))))
    assert dumeq(solveset(d * exp(exp(exp(exp(a * x + b)))) + c, x), ImageSet(Lambda(x, (-b + x) / a), ImageSet(Lambda(((k3, k2, k1, n),), I * (2 * k3 * pi + arg(I * (2 * k2 * pi + arg(I * (2 * k1 * pi + arg(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d)))) + log(Abs(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d)))))) + log(Abs(I * (2 * k1 * pi + arg(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d)))) + log(Abs(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d)))))))) + log(Abs(I * (2 * k2 * pi + arg(I * (2 * k1 * pi + arg(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d)))) + log(Abs(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d)))))) + log(Abs(I * (2 * k1 * pi + arg(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d)))) + log(Abs(I * (2 * n * pi + arg(-c / d)) + log(Abs(c / d))))))))), ProductSet(S.Integers, S.Integers, S.Integers, S.Integers))))