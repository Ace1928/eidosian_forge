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
def test_mean_var_cov(self):
    alpha = np.array([1.0, 0.8, 0.2])
    d = dirichlet(alpha)
    expected_mean = [0.5, 0.4, 0.1]
    expected_var = [1.0 / 12.0, 0.08, 0.03]
    expected_cov = [[1.0 / 12, -1.0 / 15, -1.0 / 60], [-1.0 / 15, 2.0 / 25, -1.0 / 75], [-1.0 / 60, -1.0 / 75, 3.0 / 100]]
    assert_array_almost_equal(d.mean(), expected_mean)
    assert_array_almost_equal(d.var(), expected_var)
    assert_array_almost_equal(d.cov(), expected_cov)