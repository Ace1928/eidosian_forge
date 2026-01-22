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
def test_logpdf_4x4(self):
    """Regression test for gh-8844."""
    X = np.array([[2, 1, 0, 0.5], [1, 2, 0.5, 0.5], [0, 0.5, 3, 1], [0.5, 0.5, 1, 2]])
    Psi = np.array([[9, 7, 3, 1], [7, 9, 5, 1], [3, 5, 8, 2], [1, 1, 2, 9]])
    nu = 6
    prob = invwishart.logpdf(X, nu, Psi)
    p = X.shape[0]
    sig, logdetX = np.linalg.slogdet(X)
    sig, logdetPsi = np.linalg.slogdet(Psi)
    M = np.linalg.solve(X, Psi)
    expected = nu / 2 * logdetPsi - nu * p / 2 * np.log(2) - multigammaln(nu / 2, p) - (nu + p + 1) / 2 * logdetX - 0.5 * M.trace()
    assert_allclose(prob, expected)