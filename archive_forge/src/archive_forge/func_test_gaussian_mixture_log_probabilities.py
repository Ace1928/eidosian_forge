import copy
import itertools
import re
import sys
import warnings
from io import StringIO
from unittest.mock import Mock
import numpy as np
import pytest
from scipy import linalg, stats
import sklearn
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_spd_matrix
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_gaussian_mixture_log_probabilities():
    from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_samples = 500
    n_features = rand_data.n_features
    n_components = rand_data.n_components
    means = rand_data.means
    covars_diag = rng.rand(n_components, n_features)
    X = rng.rand(n_samples, n_features)
    log_prob_naive = _naive_lmvnpdf_diag(X, means, covars_diag)
    precs_full = np.array([np.diag(1.0 / np.sqrt(x)) for x in covars_diag])
    log_prob = _estimate_log_gaussian_prob(X, means, precs_full, 'full')
    assert_array_almost_equal(log_prob, log_prob_naive)
    precs_chol_diag = 1.0 / np.sqrt(covars_diag)
    log_prob = _estimate_log_gaussian_prob(X, means, precs_chol_diag, 'diag')
    assert_array_almost_equal(log_prob, log_prob_naive)
    covars_tied = np.array([x for x in covars_diag]).mean(axis=0)
    precs_tied = np.diag(np.sqrt(1.0 / covars_tied))
    log_prob_naive = _naive_lmvnpdf_diag(X, means, [covars_tied] * n_components)
    log_prob = _estimate_log_gaussian_prob(X, means, precs_tied, 'tied')
    assert_array_almost_equal(log_prob, log_prob_naive)
    covars_spherical = covars_diag.mean(axis=1)
    precs_spherical = 1.0 / np.sqrt(covars_diag.mean(axis=1))
    log_prob_naive = _naive_lmvnpdf_diag(X, means, [[k] * n_features for k in covars_spherical])
    log_prob = _estimate_log_gaussian_prob(X, means, precs_spherical, 'spherical')
    assert_array_almost_equal(log_prob, log_prob_naive)