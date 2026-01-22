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
def test_gaussian_inverse():
    a, b = symbols('a b')
    assert GaussianInverse('x', a, b)
    a, b, z = symbols('a b z')
    X = Wald('x', a, b)
    assert density(X)(z) == sqrt(2) * sqrt(b / z ** 3) * exp(-b * (-a + z) ** 2 / (2 * a ** 2 * z)) / (2 * sqrt(pi))
    a, b = symbols('a b', positive=True)
    z = Symbol('z', positive=True)
    X = GaussianInverse('x', a, b)
    assert density(X)(z) == sqrt(2) * sqrt(b) * sqrt(z ** (-3)) * exp(-b * (-a + z) ** 2 / (2 * a ** 2 * z)) / (2 * sqrt(pi))
    assert E(X) == a
    assert variance(X).expand() == a ** 3 / b
    assert cdf(X)(z) == (S.Half - erf(sqrt(2) * sqrt(b) * (1 + z / a) / (2 * sqrt(z))) / 2) * exp(2 * b / a) + erf(sqrt(2) * sqrt(b) * (-1 + z / a) / (2 * sqrt(z))) / 2 + S.Half
    a = symbols('a', nonpositive=True)
    raises(ValueError, lambda: GaussianInverse('x', a, b))
    a = symbols('a', positive=True)
    b = symbols('b', nonpositive=True)
    raises(ValueError, lambda: GaussianInverse('x', a, b))