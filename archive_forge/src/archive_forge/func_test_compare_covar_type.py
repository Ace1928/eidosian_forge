import copy
import numpy as np
import pytest
from scipy.special import gammaln
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture._bayesian_mixture import _log_dirichlet_norm, _log_wishart_norm
from sklearn.mixture.tests.test_gaussian_mixture import RandomData
from sklearn.utils._testing import (
def test_compare_covar_type():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    X = rand_data.X['full']
    n_components = rand_data.n_components
    for prior_type in PRIOR_TYPE:
        bgmm = BayesianGaussianMixture(weight_concentration_prior_type=prior_type, n_components=2 * n_components, covariance_type='full', max_iter=1, random_state=0, tol=1e-07)
        bgmm._check_parameters(X)
        bgmm._initialize_parameters(X, np.random.RandomState(0))
        full_covariances = bgmm.covariances_ * bgmm.degrees_of_freedom_[:, np.newaxis, np.newaxis]
        bgmm = BayesianGaussianMixture(weight_concentration_prior_type=prior_type, n_components=2 * n_components, covariance_type='tied', max_iter=1, random_state=0, tol=1e-07)
        bgmm._check_parameters(X)
        bgmm._initialize_parameters(X, np.random.RandomState(0))
        tied_covariance = bgmm.covariances_ * bgmm.degrees_of_freedom_
        assert_almost_equal(tied_covariance, np.mean(full_covariances, 0))
        bgmm = BayesianGaussianMixture(weight_concentration_prior_type=prior_type, n_components=2 * n_components, covariance_type='diag', max_iter=1, random_state=0, tol=1e-07)
        bgmm._check_parameters(X)
        bgmm._initialize_parameters(X, np.random.RandomState(0))
        diag_covariances = bgmm.covariances_ * bgmm.degrees_of_freedom_[:, np.newaxis]
        assert_almost_equal(diag_covariances, np.array([np.diag(cov) for cov in full_covariances]))
        bgmm = BayesianGaussianMixture(weight_concentration_prior_type=prior_type, n_components=2 * n_components, covariance_type='spherical', max_iter=1, random_state=0, tol=1e-07)
        bgmm._check_parameters(X)
        bgmm._initialize_parameters(X, np.random.RandomState(0))
        spherical_covariances = bgmm.covariances_ * bgmm.degrees_of_freedom_
        assert_almost_equal(spherical_covariances, np.mean(diag_covariances, 1))