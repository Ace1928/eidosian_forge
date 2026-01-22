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
def test_mean_cov(self):
    P = np.diag(1 / np.array([1, 2, 3]))
    cov_object = _covariance.CovViaPrecision(P)
    message = '`cov` represents a covariance matrix in 3 dimensions...'
    with pytest.raises(ValueError, match=message):
        multivariate_normal.entropy([0, 0], cov_object)
    with pytest.raises(ValueError, match=message):
        multivariate_normal([0, 0], cov_object)
    x = [0.5, 0.5, 0.5]
    ref = multivariate_normal.pdf(x, [0, 0, 0], cov_object)
    assert_equal(multivariate_normal.pdf(x, cov=cov_object), ref)
    ref = multivariate_normal.pdf(x, [1, 1, 1], cov_object)
    assert_equal(multivariate_normal.pdf(x, 1, cov=cov_object), ref)