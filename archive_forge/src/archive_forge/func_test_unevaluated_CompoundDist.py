from sympy.concrete.summations import Sum
from sympy.core.numbers import (oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.sets.sets import Interval
from sympy.stats import (Normal, P, E, density, Gamma, Poisson, Rayleigh,
from sympy.stats.compound_rv import CompoundDistribution, CompoundPSpace
from sympy.stats.crv_types import NormalDistribution
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.frv_types import BernoulliDistribution
from sympy.testing.pytest import raises, ignore_warnings
from sympy.stats.joint_rv_types import MultivariateNormalDistribution
from sympy.abc import x
def test_unevaluated_CompoundDist():
    R = Rayleigh('R', 4)
    X = Normal('X', 3, R)
    ans = '\n        Piecewise(((-sqrt(pi)*sinh(x/4 - 3/4) + sqrt(pi)*cosh(x/4 - 3/4))/(\n        8*sqrt(pi)), Abs(arg(x - 3)) <= pi/4), (Integral(sqrt(2)*exp(-(x - 3)\n        **2/(2*R**2))*exp(-R**2/32)/(32*sqrt(pi)), (R, 0, oo)), True))'
    assert streq(density(X)(x), ans)
    expre = '\n        Integral(X*Integral(sqrt(2)*exp(-(X-3)**2/(2*R**2))*exp(-R**2/32)/(32*\n        sqrt(pi)),(R,0,oo)),(X,-oo,oo))'
    with ignore_warnings(UserWarning):
        assert streq(E(X, evaluate=False).rewrite(Integral), expre)
    X = Poisson('X', 1)
    Y = Poisson('Y', X)
    Z = Poisson('Z', Y)
    exprd = Sum(exp(-Y) * Y ** x * Sum(exp(-1) * exp(-X) * X ** Y / (factorial(X) * factorial(Y)), (X, 0, oo)) / factorial(x), (Y, 0, oo))
    assert density(Z)(x) == exprd
    N = Normal('N', 1, 2)
    M = Normal('M', 3, 4)
    D = Normal('D', M, N)
    exprd = '\n        Integral(sqrt(2)*exp(-(N-1)**2/8)*Integral(exp(-(x-M)**2/(2*N**2))*exp\n        (-(M-3)**2/32)/(8*pi*N),(M,-oo,oo))/(4*sqrt(pi)),(N,-oo,oo))'
    assert streq(density(D, evaluate=False)(x), exprd)