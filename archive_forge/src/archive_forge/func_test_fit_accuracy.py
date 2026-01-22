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
@pytest.mark.parametrize('dim', [2, 3, 6])
@pytest.mark.parametrize('kappa, mu_tol, kappa_tol', [(1, 0.05, 0.05), (10, 0.01, 0.01), (100, 0.005, 0.02), (1000, 0.001, 0.02)])
def test_fit_accuracy(self, dim, kappa, mu_tol, kappa_tol):
    mu = np.full((dim,), 1 / np.sqrt(dim))
    vmf_dist = vonmises_fisher(mu, kappa)
    rng = np.random.default_rng(2777937887058094419)
    n_samples = 10000
    samples = vmf_dist.rvs(n_samples, random_state=rng)
    mu_fit, kappa_fit = vonmises_fisher.fit(samples)
    angular_error = np.arccos(mu.dot(mu_fit))
    assert_allclose(angular_error, 0.0, atol=mu_tol, rtol=0)
    assert_allclose(kappa, kappa_fit, rtol=kappa_tol)