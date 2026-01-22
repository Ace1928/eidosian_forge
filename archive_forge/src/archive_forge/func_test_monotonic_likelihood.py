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
def test_monotonic_likelihood():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    n_components = rand_data.n_components
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        gmm = GaussianMixture(n_components=n_components, covariance_type=covar_type, reg_covar=0, warm_start=True, max_iter=1, random_state=rng, tol=1e-07)
        current_log_likelihood = -np.inf
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            for _ in range(600):
                prev_log_likelihood = current_log_likelihood
                current_log_likelihood = gmm.fit(X).score(X)
                assert current_log_likelihood >= prev_log_likelihood
                if gmm.converged_:
                    break
            assert gmm.converged_