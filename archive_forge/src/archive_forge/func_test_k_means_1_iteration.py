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
@pytest.mark.parametrize('array_constr', data_containers, ids=data_containers_ids)
@pytest.mark.parametrize('algo', ['lloyd', 'elkan'])
def test_k_means_1_iteration(array_constr, algo, global_random_seed):
    X = np.random.RandomState(global_random_seed).uniform(size=(100, 5))
    init_centers = X[:5]
    X = array_constr(X)

    def py_kmeans(X, init):
        new_centers = init.copy()
        labels = pairwise_distances_argmin(X, init)
        for label in range(init.shape[0]):
            new_centers[label] = X[labels == label].mean(axis=0)
        labels = pairwise_distances_argmin(X, new_centers)
        return (labels, new_centers)
    py_labels, py_centers = py_kmeans(X, init_centers)
    cy_kmeans = KMeans(n_clusters=5, n_init=1, init=init_centers, algorithm=algo, max_iter=1).fit(X)
    cy_labels = cy_kmeans.labels_
    cy_centers = cy_kmeans.cluster_centers_
    assert_array_equal(py_labels, cy_labels)
    assert_allclose(py_centers, cy_centers)