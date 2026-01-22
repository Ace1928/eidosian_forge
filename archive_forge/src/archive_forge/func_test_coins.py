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
def test_coins():
    C, D = (Coin('C'), Coin('D'))
    H, T = symbols('H, T')
    assert P(Eq(C, D)) == S.Half
    assert density(Tuple(C, D)) == {(H, H): Rational(1, 4), (H, T): Rational(1, 4), (T, H): Rational(1, 4), (T, T): Rational(1, 4)}
    assert dict(density(C).items()) == {H: S.Half, T: S.Half}
    F = Coin('F', Rational(1, 10))
    assert P(Eq(F, H)) == Rational(1, 10)
    d = pspace(C).domain
    assert d.as_boolean() == Or(Eq(C.symbol, H), Eq(C.symbol, T))
    raises(ValueError, lambda: P(C > D))