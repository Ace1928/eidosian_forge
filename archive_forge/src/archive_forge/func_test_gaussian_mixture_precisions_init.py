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
@pytest.mark.parametrize('covariance_type', COVARIANCE_TYPE)
def test_gaussian_mixture_precisions_init(covariance_type, global_random_seed):
    """Non-regression test for #26415."""
    X, resp = _generate_data(seed=global_random_seed, n_samples=100, n_features=3, n_components=4)
    precisions_init, desired_precisions_cholesky = _calculate_precisions(X, resp, covariance_type)
    gmm = GaussianMixture(covariance_type=covariance_type, precisions_init=precisions_init)
    gmm._initialize(X, resp)
    actual_precisions_cholesky = gmm.precisions_cholesky_
    assert_allclose(actual_precisions_cholesky, desired_precisions_cholesky)