import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.xslow()
@pytest.mark.parametrize('axis', (-1, 2))
def test_vectorized_nsamp_ptype_both(self, axis):
    rng = np.random.default_rng(6709265303529651545)
    x = rng.random(size=3)
    y = rng.random(size=(1, 3, 2))
    z = rng.random(size=(2, 1, 4))
    data = (x, y, z)

    def statistic1d(*data):
        return stats.kruskal(*data).statistic

    def pvalue1d(*data):
        return stats.kruskal(*data).pvalue
    statistic = _resampling._vectorize_statistic(statistic1d)
    pvalue = _resampling._vectorize_statistic(pvalue1d)
    x2 = np.broadcast_to(x, (2, 3, 3))
    y2 = np.broadcast_to(y, (2, 3, 2))
    z2 = np.broadcast_to(z, (2, 3, 4))
    expected_statistic = statistic(x2, y2, z2, axis=axis)
    expected_pvalue = pvalue(x2, y2, z2, axis=axis)
    kwds = {'vectorized': False, 'axis': axis, 'alternative': 'greater', 'permutation_type': 'independent', 'random_state': self.rng}
    res = permutation_test(data, statistic1d, n_resamples=np.inf, **kwds)
    res2 = permutation_test(data, statistic1d, n_resamples=1000, **kwds)
    assert_allclose(res.statistic, expected_statistic, rtol=self.rtol)
    assert_allclose(res.statistic, res2.statistic, rtol=self.rtol)
    assert_allclose(res.pvalue, expected_pvalue, atol=0.06)
    assert_allclose(res.pvalue, res2.pvalue, atol=0.03)