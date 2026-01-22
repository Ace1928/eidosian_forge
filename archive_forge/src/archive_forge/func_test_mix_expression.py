from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import (Eq, Ne)
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.stats import (Poisson, Beta, Exponential, P,
from sympy.stats.crv_types import Normal
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
from sympy.stats.joint_rv import MarginalDistribution
from sympy.stats.rv import pspace, density
from sympy.testing.pytest import ignore_warnings
def test_mix_expression():
    Y, E = (Poisson('Y', 1), Exponential('E', 1))
    k = Dummy('k')
    expr1 = Integral(Sum(exp(-1) * Integral(exp(-k) * DiracDelta(k - 2), (k, 0, oo)) / factorial(k), (k, 0, oo)), (k, -oo, 0))
    expr2 = Integral(Sum(exp(-1) * Integral(exp(-k) * DiracDelta(k - 2), (k, 0, oo)) / factorial(k), (k, 0, oo)), (k, 0, oo))
    assert P(Eq(Y + E, 1)) == 0
    assert P(Ne(Y + E, 2)) == 1
    with ignore_warnings(UserWarning):
        assert P(E + Y < 2, evaluate=False).rewrite(Integral).dummy_eq(expr1)
        assert P(E + Y > 2, evaluate=False).rewrite(Integral).dummy_eq(expr2)