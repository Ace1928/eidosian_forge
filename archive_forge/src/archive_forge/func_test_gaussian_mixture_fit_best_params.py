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
def test_gaussian_mixture_fit_best_params():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components = rand_data.n_components
    n_init = 10
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        g = GaussianMixture(n_components=n_components, n_init=1, reg_covar=0, random_state=rng, covariance_type=covar_type)
        ll = []
        for _ in range(n_init):
            g.fit(X)
            ll.append(g.score(X))
        ll = np.array(ll)
        g_best = GaussianMixture(n_components=n_components, n_init=n_init, reg_covar=0, random_state=rng, covariance_type=covar_type)
        g_best.fit(X)
        assert_almost_equal(ll.min(), g_best.score(X))