import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution
from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit
def cases_test_fit_mse():
    skip_basic_fit = {'levy_stable', 'studentized_range', 'ksone', 'skewnorm', 'norminvgauss', 'kstwo', 'geninvgauss', 'gausshyper', 'genhyperbolic', 'tukeylambda', 'vonmises'}
    slow_basic_fit = {'alpha', 'anglit', 'arcsine', 'betabinom', 'bradford', 'chi', 'chi2', 'crystalball', 'dgamma', 'dweibull', 'erlang', 'exponnorm', 'exponpow', 'exponweib', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'gamma', 'genexpon', 'genextreme', 'genhalflogistic', 'genlogistic', 'genpareto', 'gompertz', 'hypergeom', 'invweibull', 'jf_skew_t', 'johnsonsb', 'johnsonsu', 'kappa3', 'kstwobign', 'laplace_asymmetric', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'nhypergeom', 'pareto', 'powernorm', 'randint', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncpareto', 'truncweibull_min', 'uniform', 'vonmises_line', 'wald', 'weibull_max', 'weibull_min', 'wrapcauchy'}
    xslow_basic_fit = {'beta', 'betaprime', 'burr', 'burr12', 'f', 'gengamma', 'gennorm', 'halfgennorm', 'invgamma', 'invgauss', 'kappa4', 'loguniform', 'ncf', 'nchypergeom_fisher', 'nchypergeom_wallenius', 'nct', 'ncx2', 'pearson3', 'powerlaw', 'powerlognorm', 'rdist', 'reciprocal', 'rel_breitwigner', 'rice', 'trapezoid', 'truncnorm', 'zipfian'}
    warns_basic_fit = {'skellam'}
    for dist in dict(distdiscrete + distcont):
        if dist in skip_basic_fit or not isinstance(dist, str):
            reason = 'Fails. Oh well.'
            yield pytest.param(dist, marks=pytest.mark.skip(reason=reason))
        elif dist in slow_basic_fit:
            reason = 'too slow (>= 0.25s)'
            yield pytest.param(dist, marks=pytest.mark.slow(reason=reason))
        elif dist in xslow_basic_fit:
            reason = 'too slow (>= 1.0s)'
            yield pytest.param(dist, marks=pytest.mark.xslow(reason=reason))
        elif dist in warns_basic_fit:
            mark = pytest.mark.filterwarnings('ignore::RuntimeWarning')
            yield pytest.param(dist, marks=mark)
        else:
            yield dist