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
@XFAIL
def test_exponential_complex():
    n = Dummy('n')
    assert dumeq(solveset_complex(2 ** x + 4 ** x, x), imageset(Lambda(n, I * (2 * n * pi + pi) / log(2)), S.Integers))
    assert solveset_complex(x ** z * y ** z - 2, z) == FiniteSet(log(2) / (log(x) + log(y)))
    assert dumeq(solveset_complex(4 ** (x / 2) - 2 ** (x / 3), x), imageset(Lambda(n, 3 * n * I * pi / log(2)), S.Integers))
    assert dumeq(solveset(2 ** x + 32, x), imageset(Lambda(n, (I * (2 * n * pi + pi) + 5 * log(2)) / log(2)), S.Integers))
    eq = (2 ** exp(y ** 2 / x) + 2) / (x ** 2 + 15)
    a = sqrt(x) * sqrt(-log(log(2)) + log(log(2) + 2 * n * I * pi))
    assert solveset_complex(eq, y) == FiniteSet(-a, a)
    union1 = imageset(Lambda(n, I * (2 * n * pi - pi * Rational(2, 3)) / log(2)), S.Integers)
    union2 = imageset(Lambda(n, I * (2 * n * pi + pi * Rational(2, 3)) / log(2)), S.Integers)
    assert dumeq(solveset(2 ** x + 4 ** x + 8 ** x, x), Union(union1, union2))
    eq = 4 ** (x + 1) + 4 ** (x + 2) + 4 ** (x - 1) - 3 ** (x + 2) - 3 ** (x + 3)
    res = solveset(eq, x)
    num = 2 * n * I * pi - 4 * log(2) + 2 * log(3)
    den = -2 * log(2) + log(3)
    ans = imageset(Lambda(n, num / den), S.Integers)
    assert dumeq(res, ans)