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
def test_gaussian_mixture_all_init_does_not_estimate_gaussian_parameters(monkeypatch, global_random_seed):
    """When all init parameters are provided, the Gaussian parameters
    are not estimated.

    Non-regression test for gh-26015.
    """
    mock = Mock(side_effect=_estimate_gaussian_parameters)
    monkeypatch.setattr(sklearn.mixture._gaussian_mixture, '_estimate_gaussian_parameters', mock)
    rng = np.random.RandomState(global_random_seed)
    rand_data = RandomData(rng)
    gm = GaussianMixture(n_components=rand_data.n_components, weights_init=rand_data.weights, means_init=rand_data.means, precisions_init=rand_data.precisions['full'], random_state=rng)
    gm.fit(rand_data.X['full'])
    assert mock.call_count == gm.n_iter_