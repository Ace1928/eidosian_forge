import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest
from scipy import stats
from scipy.stats import sobol_indices
from scipy.stats._resampling import BootstrapResult
from scipy.stats._sensitivity_analysis import (
@pytest.mark.xfail_on_32bit("Can't create large array for test")
@pytest.mark.parametrize('func', [f_ishigami, pytest.param(f_ishigami_vec, marks=pytest.mark.slow)], ids=['scalar', 'vector'])
def test_ishigami(self, ishigami_ref_indices, func):
    rng = np.random.default_rng(28631265345463262246170309650372465332)
    res = sobol_indices(func=func, n=4096, dists=self.dists, random_state=rng)
    if func.__name__ == 'f_ishigami_vec':
        ishigami_ref_indices = [[ishigami_ref_indices[0], ishigami_ref_indices[0]], [ishigami_ref_indices[1], ishigami_ref_indices[1]]]
    assert_allclose(res.first_order, ishigami_ref_indices[0], atol=0.01)
    assert_allclose(res.total_order, ishigami_ref_indices[1], atol=0.01)
    assert res._bootstrap_result is None
    bootstrap_res = res.bootstrap(n_resamples=99)
    assert isinstance(bootstrap_res, BootstrapSobolResult)
    assert isinstance(res._bootstrap_result, BootstrapResult)
    assert res._bootstrap_result.confidence_interval.low.shape[0] == 2
    assert res._bootstrap_result.confidence_interval.low[1].shape == res.first_order.shape
    assert bootstrap_res.first_order.confidence_interval.low.shape == res.first_order.shape
    assert bootstrap_res.total_order.confidence_interval.low.shape == res.total_order.shape
    assert_array_less(bootstrap_res.first_order.confidence_interval.low, res.first_order)
    assert_array_less(res.first_order, bootstrap_res.first_order.confidence_interval.high)
    assert_array_less(bootstrap_res.total_order.confidence_interval.low, res.total_order)
    assert_array_less(res.total_order, bootstrap_res.total_order.confidence_interval.high)
    assert isinstance(res.bootstrap(confidence_level=0.9, n_resamples=99), BootstrapSobolResult)
    assert isinstance(res._bootstrap_result, BootstrapResult)