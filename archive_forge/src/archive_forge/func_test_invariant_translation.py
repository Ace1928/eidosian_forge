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
@ignore_warnings(category=ConvergenceWarning)
def test_invariant_translation():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=100)
    n_components = 2 * rand_data.n_components
    for prior_type in PRIOR_TYPE:
        for covar_type in COVARIANCE_TYPE:
            X = rand_data.X[covar_type]
            bgmm1 = BayesianGaussianMixture(weight_concentration_prior_type=prior_type, n_components=n_components, max_iter=100, random_state=0, tol=0.001, reg_covar=0).fit(X)
            bgmm2 = BayesianGaussianMixture(weight_concentration_prior_type=prior_type, n_components=n_components, max_iter=100, random_state=0, tol=0.001, reg_covar=0).fit(X + 100)
            assert_almost_equal(bgmm1.means_, bgmm2.means_ - 100)
            assert_almost_equal(bgmm1.weights_, bgmm2.weights_)
            assert_almost_equal(bgmm1.covariances_, bgmm2.covariances_)