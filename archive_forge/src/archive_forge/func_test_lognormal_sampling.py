from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.sets.sets import Interval
from sympy.external import import_module
from sympy.stats import Beta, Chi, Normal, Gamma, Exponential, LogNormal, Pareto, ChiSquared, Uniform, sample, \
from sympy.testing.pytest import skip, raises
def test_lognormal_sampling():
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    for i in range(3):
        X = LogNormal('x', i, 1)
        assert sample(X) in X.pspace.domain.set
    size = 5
    samps = sample(X, size=size)
    for samp in samps:
        assert samp in X.pspace.domain.set