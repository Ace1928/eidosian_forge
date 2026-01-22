from sympy.concrete.summations import Sum
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, nan)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.beta_functions import beta
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polytools import cancel
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import simplify
from sympy.matrices import Matrix
from sympy.stats import (DiscreteUniform, Die, Bernoulli, Coin, Binomial, BetaBinomial,
from sympy.stats.frv_types import DieDistribution, BinomialDistribution, \
from sympy.stats.rv import Density
from sympy.testing.pytest import raises
def test_dice():
    X, Y, Z = (Die('X', 6), Die('Y', 6), Die('Z', 6))
    a, b, t, p = symbols('a b t p')
    assert E(X) == 3 + S.Half
    assert variance(X) == Rational(35, 12)
    assert E(X + Y) == 7
    assert E(X + X) == 7
    assert E(a * X + b) == a * E(X) + b
    assert variance(X + Y) == variance(X) + variance(Y) == cmoment(X + Y, 2)
    assert variance(X + X) == 4 * variance(X) == cmoment(X + X, 2)
    assert cmoment(X, 0) == 1
    assert cmoment(4 * X, 3) == 64 * cmoment(X, 3)
    assert covariance(X, Y) is S.Zero
    assert covariance(X, X + Y) == variance(X)
    assert density(Eq(cos(X * S.Pi), 1))[True] == S.Half
    assert correlation(X, Y) == 0
    assert correlation(X, Y) == correlation(Y, X)
    assert smoment(X + Y, 3) == skewness(X + Y)
    assert smoment(X + Y, 4) == kurtosis(X + Y)
    assert smoment(X, 0) == 1
    assert P(X > 3) == S.Half
    assert P(2 * X > 6) == S.Half
    assert P(X > Y) == Rational(5, 12)
    assert P(Eq(X, Y)) == P(Eq(X, 1))
    assert E(X, X > 3) == 5 == moment(X, 1, 0, X > 3)
    assert E(X, Y > 3) == E(X) == moment(X, 1, 0, Y > 3)
    assert E(X + Y, Eq(X, Y)) == E(2 * X)
    assert moment(X, 0) == 1
    assert moment(5 * X, 2) == 25 * moment(X, 2)
    assert quantile(X)(p) == Piecewise((nan, (p > 1) | (p < 0)), (S.One, p <= Rational(1, 6)), (S(2), p <= Rational(1, 3)), (S(3), p <= S.Half), (S(4), p <= Rational(2, 3)), (S(5), p <= Rational(5, 6)), (S(6), p <= 1))
    assert P(X > 3, X > 3) is S.One
    assert P(X > Y, Eq(Y, 6)) is S.Zero
    assert P(Eq(X + Y, 12)) == Rational(1, 36)
    assert P(Eq(X + Y, 12), Eq(X, 6)) == Rational(1, 6)
    assert density(X + Y) == density(Y + Z) != density(X + X)
    d = density(2 * X + Y ** Z)
    assert d[S(22)] == Rational(1, 108) and d[S(4100)] == Rational(1, 216) and (S(3130) not in d)
    assert pspace(X).domain.as_boolean() == Or(*[Eq(X.symbol, i) for i in [1, 2, 3, 4, 5, 6]])
    assert where(X > 3).set == FiniteSet(4, 5, 6)
    assert characteristic_function(X)(t) == exp(6 * I * t) / 6 + exp(5 * I * t) / 6 + exp(4 * I * t) / 6 + exp(3 * I * t) / 6 + exp(2 * I * t) / 6 + exp(I * t) / 6
    assert moment_generating_function(X)(t) == exp(6 * t) / 6 + exp(5 * t) / 6 + exp(4 * t) / 6 + exp(3 * t) / 6 + exp(2 * t) / 6 + exp(t) / 6
    assert median(X) == FiniteSet(3, 4)
    D = Die('D', 7)
    assert median(D) == FiniteSet(4)
    BayesTest(X > 3, X + Y < 5)
    BayesTest(Eq(X - Y, Z), Z > Y)
    BayesTest(X > 3, X > 2)
    raises(ValueError, lambda: Die('X', -1))
    raises(ValueError, lambda: Die('X', 0))
    raises(ValueError, lambda: Die('X', 1.5))
    n, k = symbols('n, k', positive=True)
    D = Die('D', n)
    dens = density(D).dict
    assert dens == Density(DieDistribution(n))
    assert set(dens.subs(n, 4).doit().keys()) == {1, 2, 3, 4}
    assert set(dens.subs(n, 4).doit().values()) == {Rational(1, 4)}
    k = Dummy('k', integer=True)
    assert E(D).dummy_eq(Sum(Piecewise((k / n, k <= n), (0, True)), (k, 1, n)))
    assert variance(D).subs(n, 6).doit() == Rational(35, 12)
    ki = Dummy('ki')
    cumuf = cdf(D)(k)
    assert cumuf.dummy_eq(Sum(Piecewise((1 / n, (ki >= 1) & (ki <= n)), (0, True)), (ki, 1, k)))
    assert cumuf.subs({n: 6, k: 2}).doit() == Rational(1, 3)
    t = Dummy('t')
    cf = characteristic_function(D)(t)
    assert cf.dummy_eq(Sum(Piecewise((exp(ki * I * t) / n, (ki >= 1) & (ki <= n)), (0, True)), (ki, 1, n)))
    assert cf.subs(n, 3).doit() == exp(3 * I * t) / 3 + exp(2 * I * t) / 3 + exp(I * t) / 3
    mgf = moment_generating_function(D)(t)
    assert mgf.dummy_eq(Sum(Piecewise((exp(ki * t) / n, (ki >= 1) & (ki <= n)), (0, True)), (ki, 1, n)))
    assert mgf.subs(n, 3).doit() == exp(3 * t) / 3 + exp(2 * t) / 3 + exp(t) / 3