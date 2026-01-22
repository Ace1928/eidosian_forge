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
def test_bayesian_mixture_precisions_prior_initialisation():
    rng = np.random.RandomState(0)
    n_samples, n_features = (10, 2)
    X = rng.rand(n_samples, n_features)
    bad_degrees_of_freedom_prior_ = n_features - 1.0
    bgmm = BayesianGaussianMixture(degrees_of_freedom_prior=bad_degrees_of_freedom_prior_, random_state=rng)
    msg = f"The parameter 'degrees_of_freedom_prior' should be greater than {n_features - 1}, but got {bad_degrees_of_freedom_prior_:.3f}."
    with pytest.raises(ValueError, match=msg):
        bgmm.fit(X)
    degrees_of_freedom_prior = rng.rand() + n_features - 1.0
    bgmm = BayesianGaussianMixture(degrees_of_freedom_prior=degrees_of_freedom_prior, random_state=rng).fit(X)
    assert_almost_equal(degrees_of_freedom_prior, bgmm.degrees_of_freedom_prior_)
    degrees_of_freedom_prior_default = n_features
    bgmm = BayesianGaussianMixture(degrees_of_freedom_prior=degrees_of_freedom_prior_default, random_state=rng).fit(X)
    assert_almost_equal(degrees_of_freedom_prior_default, bgmm.degrees_of_freedom_prior_)
    covariance_prior = {'full': np.cov(X.T, bias=1) + 10, 'tied': np.cov(X.T, bias=1) + 5, 'diag': np.diag(np.atleast_2d(np.cov(X.T, bias=1))) + 3, 'spherical': rng.rand()}
    bgmm = BayesianGaussianMixture(random_state=rng)
    for cov_type in ['full', 'tied', 'diag', 'spherical']:
        bgmm.covariance_type = cov_type
        bgmm.covariance_prior = covariance_prior[cov_type]
        bgmm.fit(X)
        assert_almost_equal(covariance_prior[cov_type], bgmm.covariance_prior_)
    covariance_prior_default = {'full': np.atleast_2d(np.cov(X.T)), 'tied': np.atleast_2d(np.cov(X.T)), 'diag': np.var(X, axis=0, ddof=1), 'spherical': np.var(X, axis=0, ddof=1).mean()}
    bgmm = BayesianGaussianMixture(random_state=0)
    for cov_type in ['full', 'tied', 'diag', 'spherical']:
        bgmm.covariance_type = cov_type
        bgmm.fit(X)
        assert_almost_equal(covariance_prior_default[cov_type], bgmm.covariance_prior_)