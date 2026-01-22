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
@ignore_warnings(category=ConvergenceWarning)
def test_convergence_detected_with_warm_start():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components = rand_data.n_components
    X = rand_data.X['full']
    for max_iter in (1, 2, 50):
        gmm = GaussianMixture(n_components=n_components, warm_start=True, max_iter=max_iter, random_state=rng)
        for _ in range(100):
            gmm.fit(X)
            if gmm.converged_:
                break
        assert gmm.converged_
        assert max_iter >= gmm.n_iter_