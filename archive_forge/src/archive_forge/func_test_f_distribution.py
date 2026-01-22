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
def test_f_distribution():
    d1 = Symbol('d1', positive=True)
    d2 = Symbol('d2', positive=True)
    X = FDistribution('x', d1, d2)
    assert density(X)(x) == d2 ** (d2 / 2) * sqrt((d1 * x) ** d1 * (d1 * x + d2) ** (-d1 - d2)) / (x * beta(d1 / 2, d2 / 2))
    raises(NotImplementedError, lambda: moment_generating_function(X))
    d1 = Symbol('d1', nonpositive=True)
    raises(ValueError, lambda: FDistribution('x', d1, d1))
    d1 = Symbol('d1', positive=True, integer=False)
    raises(ValueError, lambda: FDistribution('x', d1, d1))
    d1 = Symbol('d1', positive=True)
    d2 = Symbol('d2', nonpositive=True)
    raises(ValueError, lambda: FDistribution('x', d1, d2))
    d2 = Symbol('d2', positive=True, integer=False)
    raises(ValueError, lambda: FDistribution('x', d1, d2))