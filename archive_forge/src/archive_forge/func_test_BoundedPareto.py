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
def test_BoundedPareto():
    L, H = symbols('L, H', negative=True)
    raises(ValueError, lambda: BoundedPareto('X', 1, L, H))
    L, H = symbols('L, H', real=False)
    raises(ValueError, lambda: BoundedPareto('X', 1, L, H))
    L, H = symbols('L, H', positive=True)
    raises(ValueError, lambda: BoundedPareto('X', -1, L, H))
    X = BoundedPareto('X', 2, L, H)
    assert X.pspace.domain.set == Interval(L, H)
    assert density(X)(x) == 2 * L ** 2 / (x ** 3 * (1 - L ** 2 / H ** 2))
    assert cdf(X)(x) == Piecewise((-H ** 2 * L ** 2 / (x ** 2 * (H ** 2 - L ** 2)) + H ** 2 / (H ** 2 - L ** 2), L <= x), (0, True))
    assert E(X).simplify() == 2 * H * L / (H + L)
    X = BoundedPareto('X', 1, 2, 4)
    assert E(X).simplify() == log(16)
    assert median(X) == FiniteSet(Rational(8, 3))
    assert variance(X).simplify() == 8 - 16 * log(2) ** 2