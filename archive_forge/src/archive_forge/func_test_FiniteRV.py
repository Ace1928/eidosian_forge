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
def test_FiniteRV():
    F = FiniteRV('F', {1: S.Half, 2: Rational(1, 4), 3: Rational(1, 4)}, check=True)
    p = Symbol('p', positive=True)
    assert dict(density(F).items()) == {S.One: S.Half, S(2): Rational(1, 4), S(3): Rational(1, 4)}
    assert P(F >= 2) == S.Half
    assert quantile(F)(p) == Piecewise((nan, p > S.One), (S.One, p <= S.Half), (S(2), p <= Rational(3, 4)), (S(3), True))
    assert pspace(F).domain.as_boolean() == Or(*[Eq(F.symbol, i) for i in [1, 2, 3]])
    assert F.pspace.domain.set == FiniteSet(1, 2, 3)
    raises(ValueError, lambda: FiniteRV('F', {1: S.Half, 2: S.Half, 3: S.Half}, check=True))
    raises(ValueError, lambda: FiniteRV('F', {1: S.Half, 2: Rational(-1, 2), 3: S.One}, check=True))
    raises(ValueError, lambda: FiniteRV('F', {1: S.One, 2: Rational(3, 2), 3: S.Zero, 4: Rational(-1, 2), 5: Rational(-3, 4), 6: Rational(-1, 4)}, check=True))
    X = FiniteRV('X', {1: 1, 2: 2})
    assert E(X) == 5
    assert P(X <= 2) + P(X > 2) != 1