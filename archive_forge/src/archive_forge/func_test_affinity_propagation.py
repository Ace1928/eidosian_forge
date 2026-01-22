import warnings
import numpy as np
import pytest
from sklearn.cluster import AffinityPropagation, affinity_propagation
from sklearn.cluster._affinity_propagation import _equal_similarities_and_preferences
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_affinity_propagation(global_random_seed, global_dtype):
    """Test consistency of the affinity propagations."""
    S = -euclidean_distances(X.astype(global_dtype, copy=False), squared=True)
    preference = np.median(S) * 10
    cluster_centers_indices, labels = affinity_propagation(S, preference=preference, random_state=global_random_seed)
    n_clusters_ = len(cluster_centers_indices)
    assert n_clusters == n_clusters_