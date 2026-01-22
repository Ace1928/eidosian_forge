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
def test_invert_modular():
    n = Dummy('n', integer=True)
    from sympy.solvers.solveset import _invert_modular as invert_modular
    assert invert_modular(Mod(sin(x), 7), S(5), n, x) == (Mod(sin(x), 7), 5)
    assert invert_modular(Mod(exp(x), 7), S(5), n, x) == (Mod(exp(x), 7), 5)
    assert invert_modular(Mod(log(x), 7), S(5), n, x) == (Mod(log(x), 7), 5)
    assert dumeq(invert_modular(Mod(x, 7), S(5), n, x), (x, ImageSet(Lambda(n, 7 * n + 5), S.Integers)))
    assert dumeq(invert_modular(Mod(x + 8, 7), S(5), n, x), (x, ImageSet(Lambda(n, 7 * n + 4), S.Integers)))
    assert invert_modular(Mod(x ** 2 + x, 7), S(5), n, x) == (Mod(x ** 2 + x, 7), 5)
    assert dumeq(invert_modular(Mod(3 * x, 7), S(5), n, x), (x, ImageSet(Lambda(n, 7 * n + 4), S.Integers)))
    assert invert_modular(Mod((x + 1) * (x + 2), 7), S(5), n, x) == (Mod((x + 1) * (x + 2), 7), 5)
    assert invert_modular(Mod(x ** 4, 7), S(5), n, x) == (x, S.EmptySet)
    assert dumeq(invert_modular(Mod(3 ** x, 4), S(3), n, x), (x, ImageSet(Lambda(n, 2 * n + 1), S.Naturals0)))
    assert dumeq(invert_modular(Mod(2 ** (x ** 2 + x + 1), 7), S(2), n, x), (x ** 2 + x + 1, ImageSet(Lambda(n, 3 * n + 1), S.Naturals0)))
    assert invert_modular(Mod(sin(x) ** 4, 7), S(5), n, x) == (x, S.EmptySet)