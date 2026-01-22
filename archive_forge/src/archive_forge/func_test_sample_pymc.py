from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.sets.sets import Interval
from sympy.external import import_module
from sympy.stats import Beta, Chi, Normal, Gamma, Exponential, LogNormal, Pareto, ChiSquared, Uniform, sample, \
from sympy.testing.pytest import skip, raises
def test_sample_pymc():
    distribs_pymc = [Beta('B', 1, 1), Cauchy('C', 1, 1), Normal('N', 0, 1), Gamma('G', 2, 7), GaussianInverse('GI', 1, 1), Exponential('E', 2), LogNormal('LN', 0, 1), Pareto('P', 1, 1), ChiSquared('CS', 2), Uniform('U', 0, 1)]
    size = 3
    pymc = import_module('pymc')
    if not pymc:
        skip('PyMC is not installed. Abort tests for _sample_pymc.')
    else:
        for X in distribs_pymc:
            samps = sample(X, size=size, library='pymc')
            for sam in samps:
                assert sam in X.pspace.domain.set
        raises(NotImplementedError, lambda: sample(Chi('C', 1), library='pymc'))