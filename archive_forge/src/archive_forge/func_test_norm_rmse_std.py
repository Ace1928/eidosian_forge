import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy.testing import (assert_, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.stats as stats
@pytest.mark.parametrize('method, expected', list(norm_rmse_std_cases.items()))
def test_norm_rmse_std(self, method, expected):
    reps, n, m = (10000, 50, 7)
    rmse_expected, std_expected = expected
    rvs = stats.norm.rvs(size=(reps, n), random_state=0)
    true_entropy = stats.norm.entropy()
    res = stats.differential_entropy(rvs, window_length=m, method=method, axis=-1)
    assert_allclose(np.sqrt(np.mean((res - true_entropy) ** 2)), rmse_expected, atol=0.005)
    assert_allclose(np.std(res), std_expected, atol=0.002)