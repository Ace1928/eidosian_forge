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
def test_compute_log_det_cholesky():
    n_features = 2
    rand_data = RandomData(np.random.RandomState(0))
    for covar_type in COVARIANCE_TYPE:
        covariance = rand_data.covariances[covar_type]
        if covar_type == 'full':
            predected_det = np.array([linalg.det(cov) for cov in covariance])
        elif covar_type == 'tied':
            predected_det = linalg.det(covariance)
        elif covar_type == 'diag':
            predected_det = np.array([np.prod(cov) for cov in covariance])
        elif covar_type == 'spherical':
            predected_det = covariance ** n_features
        expected_det = _compute_log_det_cholesky(_compute_precision_cholesky(covariance, covar_type), covar_type, n_features=n_features)
        assert_array_almost_equal(expected_det, -0.5 * np.log(predected_det))