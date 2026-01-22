import re
import sys
from io import StringIO
import numpy as np
import pytest
from scipy import sparse as sp
from sklearn.base import clone
from sklearn.cluster import KMeans, MiniBatchKMeans, k_means, kmeans_plusplus
from sklearn.cluster._k_means_common import (
from sklearn.cluster._kmeans import _labels_inertia, _mini_batch_step
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils._testing import (
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS, threadpool_limits
@pytest.mark.parametrize('init', ['random', 'k-means++', centers, lambda X, k, random_state: centers], ids=['random', 'k-means++', 'ndarray', 'callable'])
def test_minibatch_kmeans_partial_fit_init(init):
    n_init = 10 if isinstance(init, str) else 1
    km = MiniBatchKMeans(init=init, n_clusters=n_clusters, random_state=0, n_init=n_init)
    for i in range(100):
        km.partial_fit(X)
    _check_fitted_model(km)