import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('statistic', [np.mean, np.median, np.sum, np.std, np.min, np.max, 'count', lambda x: (x ** 2).sum(), lambda x: (x ** 2).sum() * 1j])
def test_dd_all(self, dtype, statistic):

    def ref_statistic(x):
        return len(x) if statistic == 'count' else statistic(x)
    rng = np.random.default_rng(3704743126639371)
    n = 10
    x = rng.random(size=n)
    i = x >= 0.5
    v = rng.random(size=n)
    if dtype is np.complex128:
        v = v + rng.random(size=n) * 1j
    stat, _, _ = binned_statistic_dd(x, v, statistic, bins=2)
    ref = np.array([ref_statistic(v[~i]), ref_statistic(v[i])])
    assert_allclose(stat, ref)
    assert stat.dtype == np.result_type(ref.dtype, np.float64)