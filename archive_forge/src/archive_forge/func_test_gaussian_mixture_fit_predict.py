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
@pytest.mark.filterwarnings('ignore:.*did not converge.*')
@pytest.mark.parametrize('seed, max_iter, tol', [(0, 2, 1e-07), (1, 2, 0.1), (3, 300, 1e-07), (4, 300, 0.1)])
def test_gaussian_mixture_fit_predict(seed, max_iter, tol):
    rng = np.random.RandomState(seed)
    rand_data = RandomData(rng)
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        Y = rand_data.Y
        g = GaussianMixture(n_components=rand_data.n_components, random_state=rng, weights_init=rand_data.weights, means_init=rand_data.means, precisions_init=rand_data.precisions[covar_type], covariance_type=covar_type, max_iter=max_iter, tol=tol)
        f = copy.deepcopy(g)
        Y_pred1 = f.fit(X).predict(X)
        Y_pred2 = g.fit_predict(X)
        assert_array_equal(Y_pred1, Y_pred2)
        assert adjusted_rand_score(Y, Y_pred2) > 0.95