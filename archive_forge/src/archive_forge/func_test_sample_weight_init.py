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
@pytest.mark.parametrize('init', ['k-means++', 'random'])
def test_sample_weight_init(init, global_random_seed):
    """Check that sample weight is used during init.

    `_init_centroids` is shared across all classes inheriting from _BaseKMeans so
    it's enough to check for KMeans.
    """
    rng = np.random.RandomState(global_random_seed)
    X, _ = make_blobs(n_samples=200, n_features=10, centers=10, random_state=global_random_seed)
    x_squared_norms = row_norms(X, squared=True)
    kmeans = KMeans()
    clusters_weighted = kmeans._init_centroids(X=X, x_squared_norms=x_squared_norms, init=init, sample_weight=rng.uniform(size=X.shape[0]), n_centroids=5, random_state=np.random.RandomState(global_random_seed))
    clusters = kmeans._init_centroids(X=X, x_squared_norms=x_squared_norms, init=init, sample_weight=np.ones(X.shape[0]), n_centroids=5, random_state=np.random.RandomState(global_random_seed))
    with pytest.raises(AssertionError):
        assert_allclose(clusters_weighted, clusters)