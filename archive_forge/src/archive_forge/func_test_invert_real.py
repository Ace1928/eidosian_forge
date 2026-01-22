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
def test_invert_real():
    x = Symbol('x', real=True)

    def ireal(x, s=S.Reals):
        return Intersection(s, x)
    assert invert_real(exp(x), z, x) == (x, ireal(FiniteSet(log(z))))
    y = Symbol('y', positive=True)
    n = Symbol('n', real=True)
    assert invert_real(x + 3, y, x) == (x, FiniteSet(y - 3))
    assert invert_real(x * 3, y, x) == (x, FiniteSet(y / 3))
    assert invert_real(exp(x), y, x) == (x, FiniteSet(log(y)))
    assert invert_real(exp(3 * x), y, x) == (x, FiniteSet(log(y) / 3))
    assert invert_real(exp(x + 3), y, x) == (x, FiniteSet(log(y) - 3))
    assert invert_real(exp(x) + 3, y, x) == (x, ireal(FiniteSet(log(y - 3))))
    assert invert_real(exp(x) * 3, y, x) == (x, FiniteSet(log(y / 3)))
    assert invert_real(log(x), y, x) == (x, FiniteSet(exp(y)))
    assert invert_real(log(3 * x), y, x) == (x, FiniteSet(exp(y) / 3))
    assert invert_real(log(x + 3), y, x) == (x, FiniteSet(exp(y) - 3))
    assert invert_real(Abs(x), y, x) == (x, FiniteSet(y, -y))
    assert invert_real(2 ** x, y, x) == (x, FiniteSet(log(y) / log(2)))
    assert invert_real(2 ** exp(x), y, x) == (x, ireal(FiniteSet(log(log(y) / log(2)))))
    assert invert_real(x ** 2, y, x) == (x, FiniteSet(sqrt(y), -sqrt(y)))
    assert invert_real(x ** S.Half, y, x) == (x, FiniteSet(y ** 2))
    raises(ValueError, lambda: invert_real(x, x, x))
    assert invert_real(x ** pi, y, x) == (x, FiniteSet(y ** (1 / pi)))
    assert invert_real(x ** pi, -E, x) == (x, S.EmptySet)
    assert invert_real(x ** Rational(3 / 2), 1000, x) == (x, FiniteSet(100))
    assert invert_real(x ** 1.0, 1, x) == (x ** 1.0, FiniteSet(1))
    raises(ValueError, lambda: invert_real(S.One, y, x))
    assert invert_real(x ** 31 + x, y, x) == (x ** 31 + x, FiniteSet(y))
    lhs = x ** 31 + x
    base_values = FiniteSet(y - 1, -y - 1)
    assert invert_real(Abs(x ** 31 + x + 1), y, x) == (lhs, base_values)
    assert dumeq(invert_real(sin(x), y, x), (x, imageset(Lambda(n, n * pi + (-1) ** n * asin(y)), S.Integers)))
    assert dumeq(invert_real(sin(exp(x)), y, x), (x, imageset(Lambda(n, log((-1) ** n * asin(y) + n * pi)), S.Integers)))
    assert dumeq(invert_real(csc(x), y, x), (x, imageset(Lambda(n, n * pi + (-1) ** n * acsc(y)), S.Integers)))
    assert dumeq(invert_real(csc(exp(x)), y, x), (x, imageset(Lambda(n, log((-1) ** n * acsc(y) + n * pi)), S.Integers)))
    assert dumeq(invert_real(cos(x), y, x), (x, Union(imageset(Lambda(n, 2 * n * pi + acos(y)), S.Integers), imageset(Lambda(n, 2 * n * pi - acos(y)), S.Integers))))
    assert dumeq(invert_real(cos(exp(x)), y, x), (x, Union(imageset(Lambda(n, log(2 * n * pi + acos(y))), S.Integers), imageset(Lambda(n, log(2 * n * pi - acos(y))), S.Integers))))
    assert dumeq(invert_real(sec(x), y, x), (x, Union(imageset(Lambda(n, 2 * n * pi + asec(y)), S.Integers), imageset(Lambda(n, 2 * n * pi - asec(y)), S.Integers))))
    assert dumeq(invert_real(sec(exp(x)), y, x), (x, Union(imageset(Lambda(n, log(2 * n * pi + asec(y))), S.Integers), imageset(Lambda(n, log(2 * n * pi - asec(y))), S.Integers))))
    assert dumeq(invert_real(tan(x), y, x), (x, imageset(Lambda(n, n * pi + atan(y)), S.Integers)))
    assert dumeq(invert_real(tan(exp(x)), y, x), (x, imageset(Lambda(n, log(n * pi + atan(y))), S.Integers)))
    assert dumeq(invert_real(cot(x), y, x), (x, imageset(Lambda(n, n * pi + acot(y)), S.Integers)))
    assert dumeq(invert_real(cot(exp(x)), y, x), (x, imageset(Lambda(n, log(n * pi + acot(y))), S.Integers)))
    assert dumeq(invert_real(tan(tan(x)), y, x), (tan(x), imageset(Lambda(n, n * pi + atan(y)), S.Integers)))
    x = Symbol('x', positive=True)
    assert invert_real(x ** pi, y, x) == (x, FiniteSet(y ** (1 / pi)))