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
def test_integer_domain_relational():
    eq1 = 2 * x + 3 > 0
    eq2 = x ** 2 + 3 * x - 2 >= 0
    eq3 = x + 1 / x > -2 + 1 / x
    eq4 = x + sqrt(x ** 2 - 5) > 0
    eq = x + 1 / x > -2 + 1 / x
    eq5 = eq.subs(x, log(x))
    eq6 = log(x) / x <= 0
    eq7 = log(x) / x < 0
    eq8 = x / (x - 3) < 3
    eq9 = x / (x ** 2 - 3) < 3
    assert solveset(eq1, x, S.Integers) == Range(-1, oo, 1)
    assert solveset(eq2, x, S.Integers) == Union(Range(-oo, -3, 1), Range(1, oo, 1))
    assert solveset(eq3, x, S.Integers) == Union(Range(-1, 0, 1), Range(1, oo, 1))
    assert solveset(eq4, x, S.Integers) == Range(3, oo, 1)
    assert solveset(eq5, x, S.Integers) == Range(2, oo, 1)
    assert solveset(eq6, x, S.Integers) == Range(1, 2, 1)
    assert solveset(eq7, x, S.Integers) == S.EmptySet
    assert solveset(eq8, x, domain=Range(0, 5)) == Range(0, 3, 1)
    assert solveset(eq9, x, domain=Range(0, 5)) == Union(Range(0, 2, 1), Range(2, 5, 1))
    assert solveset(x + 2 < 0, x, S.Integers) == Range(-oo, -2, 1)