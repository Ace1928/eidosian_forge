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
@pytest.mark.parametrize('x, m, n, expected', [([5], [5], 5, 1), ([3, 4], [5, 10], 7, 0.3263403), ([[[3, 5], [0, 8]], [[-1, 9], [1, 1]]], [5, 10], [[8, 8], [8, 2]], [[0.3916084, 0.006993007], [0, 0.4761905]]), (np.array([], dtype=int), np.array([], dtype=int), 0, []), ([1, 2], [4, 5], 5, 0), ([3, 3, 0], [5, 6, 7], 6, 0.01077354)])
def test_pmf(self, x, m, n, expected):
    vals = multivariate_hypergeom.pmf(x, m, n)
    assert_allclose(vals, expected, rtol=1e-07)