import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
@pytest.mark.parametrize('lmbda', [0, 0.1, 0.5, 2])
def test_lmbda_None(self, lmbda):

    def _inverse_transform(x, lmbda):
        x_inv = np.zeros(x.shape, dtype=x.dtype)
        pos = x >= 0
        if abs(lmbda) < np.spacing(1.0):
            x_inv[pos] = np.exp(x[pos]) - 1
        else:
            x_inv[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1
        if abs(lmbda - 2) > np.spacing(1.0):
            x_inv[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1, 1 / (2 - lmbda))
        else:
            x_inv[~pos] = 1 - np.exp(-x[~pos])
        return x_inv
    n_samples = 20000
    np.random.seed(1234567)
    x = np.random.normal(loc=0, scale=1, size=n_samples)
    x_inv = _inverse_transform(x, lmbda)
    xt, maxlog = stats.yeojohnson(x_inv)
    assert_allclose(maxlog, lmbda, atol=0.01)
    assert_almost_equal(0, np.linalg.norm(x - xt) / n_samples, decimal=2)
    assert_almost_equal(0, xt.mean(), decimal=1)
    assert_almost_equal(1, xt.std(), decimal=1)