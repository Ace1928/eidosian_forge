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
def test_check_precisions():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components, n_features = (rand_data.n_components, rand_data.n_features)
    precisions_bad_shape = {'full': np.ones((n_components + 1, n_features, n_features)), 'tied': np.ones((n_features + 1, n_features + 1)), 'diag': np.ones((n_components + 1, n_features)), 'spherical': np.ones(n_components + 1)}
    precisions_not_pos = np.ones((n_components, n_features, n_features))
    precisions_not_pos[0] = np.eye(n_features)
    precisions_not_pos[0, 0, 0] = -1.0
    precisions_not_positive = {'full': precisions_not_pos, 'tied': precisions_not_pos[0], 'diag': np.full((n_components, n_features), -1.0), 'spherical': np.full(n_components, -1.0)}
    not_positive_errors = {'full': 'symmetric, positive-definite', 'tied': 'symmetric, positive-definite', 'diag': 'positive', 'spherical': 'positive'}
    for covar_type in COVARIANCE_TYPE:
        X = RandomData(rng).X[covar_type]
        g = GaussianMixture(n_components=n_components, covariance_type=covar_type, random_state=rng)
        g.precisions_init = precisions_bad_shape[covar_type]
        msg = f"The parameter '{covar_type} precision' should have the shape of"
        with pytest.raises(ValueError, match=msg):
            g.fit(X)
        g.precisions_init = precisions_not_positive[covar_type]
        msg = f"'{covar_type} precision' should be {not_positive_errors[covar_type]}"
        with pytest.raises(ValueError, match=msg):
            g.fit(X)
        g.precisions_init = rand_data.precisions[covar_type]
        g.fit(X)
        assert_array_equal(rand_data.precisions[covar_type], g.precisions_init)