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
@pytest.mark.parametrize('dist, params', [(stats.norm, (0.5, 2.5)), (stats.binom, (10, 0.3, 2))])
def test_nnlf_and_related_methods(dist, params):
    rng = np.random.default_rng(983459824)
    if hasattr(dist, 'pdf'):
        logpxf = dist.logpdf
    else:
        logpxf = dist.logpmf
    x = dist.rvs(*params, size=100, random_state=rng)
    ref = -logpxf(x, *params).sum()
    res1 = dist.nnlf(params, x)
    res2 = dist._penalized_nnlf(params, x)
    assert_allclose(res1, ref)
    assert_allclose(res2, ref)