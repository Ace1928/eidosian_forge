from sympy.concrete.summations import Sum
from sympy.core.function import (Lambda, diff, expand_func)
from sympy.core.mul import Mul
from sympy.core import EulerGamma
from sympy.core.numbers import (E as e, I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (asin, atan, cos, sin, tan)
from sympy.functions.special.bessel import (besseli, besselj, besselk)
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.error_functions import (erf, erfc, erfi, expint)
from sympy.functions.special.gamma_functions import (gamma, lowergamma, uppergamma)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.sets.sets import Interval
from sympy.simplify.simplify import simplify
from sympy.utilities.lambdify import lambdify
from sympy.functions.special.error_functions import erfinv
from sympy.functions.special.hyper import meijerg
from sympy.sets.sets import FiniteSet, Complement, Intersection
from sympy.stats import (P, E, where, density, variance, covariance, skewness, kurtosis, median,
from sympy.stats.crv_types import NormalDistribution, ExponentialDistribution, ContinuousDistributionHandmade
from sympy.stats.joint_rv_types import MultivariateLaplaceDistribution, MultivariateNormalDistribution
from sympy.stats.crv import SingleContinuousPSpace, SingleContinuousDomain
from sympy.stats.compound_rv import CompoundPSpace
from sympy.stats.symbolic_probability import Probability
from sympy.testing.pytest import raises, XFAIL, slow, ignore_warnings
from sympy.core.random import verify_numerically as tn
def test_multiple_normal():
    X, Y = (Normal('x', 0, 1), Normal('y', 0, 1))
    p = Symbol('p', positive=True)
    assert E(X + Y) == 0
    assert variance(X + Y) == 2
    assert variance(X + X) == 4
    assert covariance(X, Y) == 0
    assert covariance(2 * X + Y, -X) == -2 * variance(X)
    assert skewness(X) == 0
    assert skewness(X + Y) == 0
    assert kurtosis(X) == 3
    assert kurtosis(X + Y) == 3
    assert correlation(X, Y) == 0
    assert correlation(X, X + Y) == correlation(X, X - Y)
    assert moment(X, 2) == 1
    assert cmoment(X, 3) == 0
    assert moment(X + Y, 4) == 12
    assert cmoment(X, 2) == variance(X)
    assert smoment(X * X, 2) == 1
    assert smoment(X + Y, 3) == skewness(X + Y)
    assert smoment(X + Y, 4) == kurtosis(X + Y)
    assert E(X, Eq(X + Y, 0)) == 0
    assert variance(X, Eq(X + Y, 0)) == S.Half
    assert quantile(X)(p) == sqrt(2) * erfinv(2 * p - S.One)