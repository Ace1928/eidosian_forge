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
def test_max_iter_zero():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=5)
    n_components = rand_data.n_components
    X = rand_data.X['full']
    means_init = [[20, 30], [30, 25]]
    gmm = GaussianMixture(n_components=n_components, random_state=rng, means_init=means_init, tol=1e-06, max_iter=0)
    gmm.fit(X)
    assert_allclose(gmm.means_, means_init)