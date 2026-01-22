from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.function import (Lambda, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, atan2,
from sympy.functions.special.bessel import (besselj, jn)
from sympy.functions.special.delta_functions import Heaviside
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (cancel, gcd)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.series.series import series
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.core.relational import Ne, Eq
from sympy.functions.elementary.piecewise import Piecewise
from sympy.sets.setexpr import SetExpr
from sympy.testing.pytest import XFAIL, slow, raises
def test_atan_nseries():
    assert atan(x + 2 * I)._eval_nseries(x, 4, None, 1) == I * atanh(2) - x / 3 - 2 * I * x ** 2 / 9 + 13 * x ** 3 / 81 + O(x ** 4)
    assert atan(x + 2 * I)._eval_nseries(x, 4, None, -1) == I * atanh(2) - pi - x / 3 - 2 * I * x ** 2 / 9 + 13 * x ** 3 / 81 + O(x ** 4)
    assert atan(x - 2 * I)._eval_nseries(x, 4, None, 1) == -I * atanh(2) + pi - x / 3 + 2 * I * x ** 2 / 9 + 13 * x ** 3 / 81 + O(x ** 4)
    assert atan(x - 2 * I)._eval_nseries(x, 4, None, -1) == -I * atanh(2) - x / 3 + 2 * I * x ** 2 / 9 + 13 * x ** 3 / 81 + O(x ** 4)
    assert atan(1 / x)._eval_nseries(x, 2, None, 1) == pi / 2 - x + O(x ** 2)
    assert atan(1 / x)._eval_nseries(x, 2, None, -1) == -pi / 2 - x + O(x ** 2)
    assert atan(x + I)._eval_nseries(x, 4, None) == I * log(2) / 2 + pi / 4 - I * log(x) / 2 + x / 4 + I * x ** 2 / 16 - x ** 3 / 48 + O(x ** 4)
    assert atan(x - I)._eval_nseries(x, 4, None) == -I * log(2) / 2 + pi / 4 + I * log(x) / 2 + x / 4 - I * x ** 2 / 16 - x ** 3 / 48 + O(x ** 4)