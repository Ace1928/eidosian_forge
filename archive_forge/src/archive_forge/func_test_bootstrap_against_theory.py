import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.xfail_on_32bit('MemoryError with BCa observed in CI')
@pytest.mark.parametrize('method', ['basic', 'percentile', 'BCa'])
def test_bootstrap_against_theory(method):
    rng = np.random.default_rng(2442101192988600726)
    data = stats.norm.rvs(loc=5, scale=2, size=5000, random_state=rng)
    alpha = 0.95
    dist = stats.t(df=len(data) - 1, loc=np.mean(data), scale=stats.sem(data))
    expected_interval = dist.interval(confidence=alpha)
    expected_se = dist.std()
    config = dict(data=(data,), statistic=np.mean, n_resamples=5000, method=method, random_state=rng)
    res = bootstrap(**config, confidence_level=alpha)
    assert_allclose(res.confidence_interval, expected_interval, rtol=0.0005)
    assert_allclose(res.standard_error, expected_se, atol=0.0003)
    config.update(dict(n_resamples=0, bootstrap_result=res))
    res = bootstrap(**config, confidence_level=alpha, alternative='less')
    assert_allclose(res.confidence_interval.high, dist.ppf(alpha), rtol=0.0005)
    config.update(dict(n_resamples=0, bootstrap_result=res))
    res = bootstrap(**config, confidence_level=alpha, alternative='greater')
    assert_allclose(res.confidence_interval.low, dist.ppf(1 - alpha), rtol=0.0005)