import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.xfail_on_32bit('Sensible to machine precision')
@pytest.mark.parametrize('method', ['basic', 'percentile', 'BCa'])
def test_bootstrap_alternative(method):
    rng = np.random.default_rng(5894822712842015040)
    dist = stats.norm(loc=2, scale=4)
    data = (dist.rvs(size=100, random_state=rng),)
    config = dict(data=data, statistic=np.std, random_state=rng, axis=-1)
    t = stats.bootstrap(**config, confidence_level=0.9)
    config.update(dict(n_resamples=0, bootstrap_result=t))
    l = stats.bootstrap(**config, confidence_level=0.95, alternative='less')
    g = stats.bootstrap(**config, confidence_level=0.95, alternative='greater')
    assert_equal(l.confidence_interval.high, t.confidence_interval.high)
    assert_equal(g.confidence_interval.low, t.confidence_interval.low)
    assert np.isneginf(l.confidence_interval.low)
    assert np.isposinf(g.confidence_interval.high)
    with pytest.raises(ValueError, match='`alternative` must be one of'):
        stats.bootstrap(**config, alternative='ekki-ekki')