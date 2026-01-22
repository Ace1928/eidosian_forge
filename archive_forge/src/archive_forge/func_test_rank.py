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
def test_rank(self):
    np.random.seed(1234)
    n = 4
    mean = np.random.randn(n)
    for expected_rank in range(1, n + 1):
        s = np.random.randn(n, expected_rank)
        cov = np.dot(s, s.T)
        distn = multivariate_normal(mean, cov, allow_singular=True)
        assert_equal(distn.cov_object.rank, expected_rank)