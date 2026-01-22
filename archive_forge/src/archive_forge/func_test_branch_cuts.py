from itertools import product
from sympy.concrete.summations import Sum
from sympy.core.function import (Function, diff)
from sympy.core import EulerGamma
from sympy.core.numbers import (E, I, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (binomial, factorial, subfactorial)
from sympy.functions.elementary.complexes import (Abs, re, sign)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, acsch, asech, atanh, sinh, tanh)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (cbrt, real_root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin,
from sympy.functions.special.bessel import (besseli, bessely, besselj, besselk)
from sympy.functions.special.error_functions import (Ei, erf, erfc, erfi, fresnelc, fresnels)
from sympy.functions.special.gamma_functions import (digamma, gamma, uppergamma)
from sympy.functions.special.hyper import meijerg
from sympy.integrals.integrals import (Integral, integrate)
from sympy.series.limits import (Limit, limit)
from sympy.simplify.simplify import (logcombine, simplify)
from sympy.simplify.hyperexpand import hyperexpand
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.mul import Mul
from sympy.series.limits import heuristics
from sympy.series.order import Order
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, y, z, k
def test_branch_cuts():
    assert limit(asin(I * x + 2), x, 0) == pi - asin(2)
    assert limit(asin(I * x + 2), x, 0, '-') == asin(2)
    assert limit(asin(I * x - 2), x, 0) == -asin(2)
    assert limit(asin(I * x - 2), x, 0, '-') == -pi + asin(2)
    assert limit(acos(I * x + 2), x, 0) == -acos(2)
    assert limit(acos(I * x + 2), x, 0, '-') == acos(2)
    assert limit(acos(I * x - 2), x, 0) == acos(-2)
    assert limit(acos(I * x - 2), x, 0, '-') == 2 * pi - acos(-2)
    assert limit(atan(x + 2 * I), x, 0) == I * atanh(2)
    assert limit(atan(x + 2 * I), x, 0, '-') == -pi + I * atanh(2)
    assert limit(atan(x - 2 * I), x, 0) == pi - I * atanh(2)
    assert limit(atan(x - 2 * I), x, 0, '-') == -I * atanh(2)
    assert limit(atan(1 / x), x, 0) == pi / 2
    assert limit(atan(1 / x), x, 0, '-') == -pi / 2
    assert limit(atan(x), x, oo) == pi / 2
    assert limit(atan(x), x, -oo) == -pi / 2
    assert limit(acot(x + S(1) / 2 * I), x, 0) == pi - I * acoth(S(1) / 2)
    assert limit(acot(x + S(1) / 2 * I), x, 0, '-') == -I * acoth(S(1) / 2)
    assert limit(acot(x - S(1) / 2 * I), x, 0) == I * acoth(S(1) / 2)
    assert limit(acot(x - S(1) / 2 * I), x, 0, '-') == -pi + I * acoth(S(1) / 2)
    assert limit(acot(x), x, 0) == pi / 2
    assert limit(acot(x), x, 0, '-') == -pi / 2
    assert limit(asec(I * x + S(1) / 2), x, 0) == asec(S(1) / 2)
    assert limit(asec(I * x + S(1) / 2), x, 0, '-') == -asec(S(1) / 2)
    assert limit(asec(I * x - S(1) / 2), x, 0) == 2 * pi - asec(-S(1) / 2)
    assert limit(asec(I * x - S(1) / 2), x, 0, '-') == asec(-S(1) / 2)
    assert limit(acsc(I * x + S(1) / 2), x, 0) == acsc(S(1) / 2)
    assert limit(acsc(I * x + S(1) / 2), x, 0, '-') == pi - acsc(S(1) / 2)
    assert limit(acsc(I * x - S(1) / 2), x, 0) == -pi + acsc(S(1) / 2)
    assert limit(acsc(I * x - S(1) / 2), x, 0, '-') == -acsc(S(1) / 2)
    assert limit(log(I * x - 1), x, 0) == I * pi
    assert limit(log(I * x - 1), x, 0, '-') == -I * pi
    assert limit(log(-I * x - 1), x, 0) == -I * pi
    assert limit(log(-I * x - 1), x, 0, '-') == I * pi
    assert limit(sqrt(I * x - 1), x, 0) == I
    assert limit(sqrt(I * x - 1), x, 0, '-') == -I
    assert limit(sqrt(-I * x - 1), x, 0) == -I
    assert limit(sqrt(-I * x - 1), x, 0, '-') == I
    assert limit(cbrt(I * x - 1), x, 0) == (-1) ** (S(1) / 3)
    assert limit(cbrt(I * x - 1), x, 0, '-') == -(-1) ** (S(2) / 3)
    assert limit(cbrt(-I * x - 1), x, 0) == -(-1) ** (S(2) / 3)
    assert limit(cbrt(-I * x - 1), x, 0, '-') == (-1) ** (S(1) / 3)