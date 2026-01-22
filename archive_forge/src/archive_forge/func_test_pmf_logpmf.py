import pickle
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from .test_continuous_basic import check_distribution_rvs
import numpy
import numpy as np
import scipy.linalg
from scipy.stats._multivariate import (_PSD,
from scipy.stats import (multivariate_normal, multivariate_hypergeom,
from scipy.stats import _covariance, Covariance
from scipy import stats
from scipy.integrate import romb, qmc_quad, tplquad
from scipy.special import multigammaln
from scipy._lib._pep440 import Version
from .common_tests import check_random_state_property
from .data._mvt import _qsimvtv
from unittest.mock import patch
@pytest.mark.parametrize('frozen', (True, False))
@pytest.mark.parametrize('log', (True, False))
def test_pmf_logpmf(self, frozen, log):
    rng = self.get_rng()
    row = [2, 6]
    col = [1, 3, 4]
    rvs = random_table.rvs(row, col, size=1000, method='boyett', random_state=rng)
    obj = random_table(row, col) if frozen else random_table
    method = getattr(obj, 'logpmf' if log else 'pmf')
    if not frozen:
        original_method = method

        def method(x):
            return original_method(x, row, col)
    pmf = (lambda x: np.exp(method(x))) if log else method
    unique_rvs, counts = np.unique(rvs, axis=0, return_counts=True)
    p = pmf(unique_rvs)
    assert_allclose(p * len(rvs), counts, rtol=0.1)
    p2 = pmf(list(unique_rvs[0]))
    assert_equal(p2, p[0])
    rvs_nd = rvs.reshape((10, 100) + rvs.shape[1:])
    p = pmf(rvs_nd)
    assert p.shape == (10, 100)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            pij = p[i, j]
            rvij = rvs_nd[i, j]
            qij = pmf(rvij)
            assert_equal(pij, qij)
    x = [[0, 1, 1], [2, 1, 3]]
    assert_equal(np.sum(x, axis=-1), row)
    p = pmf(x)
    assert p == 0
    x = [[0, 1, 2], [1, 2, 2]]
    assert_equal(np.sum(x, axis=-2), col)
    p = pmf(x)
    assert p == 0
    message = '`x` must be at least two-dimensional'
    with pytest.raises(ValueError, match=message):
        pmf([1])
    message = '`x` must contain only integral values'
    with pytest.raises(ValueError, match=message):
        pmf([[1.1]])
    message = '`x` must contain only integral values'
    with pytest.raises(ValueError, match=message):
        pmf([[np.nan]])
    message = '`x` must contain only non-negative values'
    with pytest.raises(ValueError, match=message):
        pmf([[-1]])
    message = 'shape of `x` must agree with `row`'
    with pytest.raises(ValueError, match=message):
        pmf([[1, 2, 3]])
    message = 'shape of `x` must agree with `col`'
    with pytest.raises(ValueError, match=message):
        pmf([[1, 2], [3, 4]])