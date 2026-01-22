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
def test_issue_12769():
    r, z, x = symbols('r z x', real=True)
    a, b, s0, K, F0, s, T = symbols('a b s0 K F0 s T', positive=True, real=True)
    fx = (F0 ** b * K ** b * r * s0 - sqrt((F0 ** 2 * K ** (2 * b) * a ** 2 * (b - 1) + F0 ** (2 * b) * K ** 2 * a ** 2 * (b - 1) + F0 ** (2 * b) * K ** (2 * b) * s0 ** 2 * (b - 1) * (b ** 2 - 2 * b + 1) - 2 * F0 ** (2 * b) * K ** (b + 1) * a * r * s0 * (b ** 2 - 2 * b + 1) + 2 * F0 ** (b + 1) * K ** (2 * b) * a * r * s0 * (b ** 2 - 2 * b + 1) - 2 * F0 ** (b + 1) * K ** (b + 1) * a ** 2 * (b - 1)) / ((b - 1) * (b ** 2 - 2 * b + 1)))) * (b * r - b - r + 1)
    assert fx.subs(K, F0).factor(deep=True) == limit(fx, K, F0).factor(deep=True)