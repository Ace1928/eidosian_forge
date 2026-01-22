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
@pytest.mark.parametrize('x, m, n, expected', [([3, 4], [5, 10], 7, -1.119814), ([3, 4], [5, 10], 0, -np.inf), ([-3, 4], [5, 10], 7, -np.inf), ([3, 4], [-5, 10], 7, np.nan), ([[1, 2], [3, 4]], [[-4, -6], [-5, -10]], [3, 7], [np.nan, np.nan]), ([-3, 4], [-5, 10], 1, np.nan), ([1, 11], [10, 1], 12, np.nan), ([1, 11], [10, -1], 12, np.nan), ([3, 4], [5, 10], -7, np.nan), ([3, 3], [5, 10], 7, -np.inf)])
def test_logpmf(self, x, m, n, expected):
    vals = multivariate_hypergeom.logpmf(x, m, n)
    assert_allclose(vals, expected, rtol=1e-06)