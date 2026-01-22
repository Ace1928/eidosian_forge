import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_bootstrap_min():
    rng = np.random.default_rng(1891289180021102)
    dist = stats.norm(loc=2, scale=4)
    data = dist.rvs(size=100, random_state=rng)
    true_min = np.min(data)
    data = (data,)
    res = bootstrap(data, np.min, method='BCa', n_resamples=100, random_state=np.random.default_rng(3942))
    assert true_min == res.confidence_interval.low
    res2 = bootstrap(-np.array(data), np.max, method='BCa', n_resamples=100, random_state=np.random.default_rng(3942))
    assert_allclose(-res.confidence_interval.low, res2.confidence_interval.high)
    assert_allclose(-res.confidence_interval.high, res2.confidence_interval.low)