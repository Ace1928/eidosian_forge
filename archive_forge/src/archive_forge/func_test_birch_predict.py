import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin, v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_birch_predict(global_random_seed, global_dtype):
    rng = np.random.RandomState(global_random_seed)
    X = generate_clustered_data(n_clusters=3, n_features=3, n_samples_per_cluster=10)
    X = X.astype(global_dtype, copy=False)
    shuffle_indices = np.arange(30)
    rng.shuffle(shuffle_indices)
    X_shuffle = X[shuffle_indices, :]
    brc = Birch(n_clusters=4, threshold=1.0)
    brc.fit(X_shuffle)
    assert brc.subcluster_centers_.dtype == global_dtype
    assert_array_equal(brc.labels_, brc.predict(X_shuffle))
    centroids = brc.subcluster_centers_
    nearest_centroid = brc.subcluster_labels_[pairwise_distances_argmin(X_shuffle, centroids)]
    assert_allclose(v_measure_score(nearest_centroid, brc.labels_), 1.0)