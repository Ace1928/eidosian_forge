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
@pytest.mark.parametrize('X_csr', X_as_any_csr)
def test_minibatch_update_consistency(X_csr, global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    centers_old = centers + rng.normal(size=centers.shape)
    centers_old_csr = centers_old.copy()
    centers_new = np.zeros_like(centers_old)
    centers_new_csr = np.zeros_like(centers_old_csr)
    weight_sums = np.zeros(centers_old.shape[0], dtype=X.dtype)
    weight_sums_csr = np.zeros(centers_old.shape[0], dtype=X.dtype)
    sample_weight = np.ones(X.shape[0], dtype=X.dtype)
    X_mb = X[:10]
    X_mb_csr = X_csr[:10]
    sample_weight_mb = sample_weight[:10]
    old_inertia = _mini_batch_step(X_mb, sample_weight_mb, centers_old, centers_new, weight_sums, np.random.RandomState(global_random_seed), random_reassign=False)
    assert old_inertia > 0.0
    labels, new_inertia = _labels_inertia(X_mb, sample_weight_mb, centers_new)
    assert new_inertia > 0.0
    assert new_inertia < old_inertia
    old_inertia_csr = _mini_batch_step(X_mb_csr, sample_weight_mb, centers_old_csr, centers_new_csr, weight_sums_csr, np.random.RandomState(global_random_seed), random_reassign=False)
    assert old_inertia_csr > 0.0
    labels_csr, new_inertia_csr = _labels_inertia(X_mb_csr, sample_weight_mb, centers_new_csr)
    assert new_inertia_csr > 0.0
    assert new_inertia_csr < old_inertia_csr
    assert_array_equal(labels, labels_csr)
    assert_allclose(centers_new, centers_new_csr)
    assert_allclose(old_inertia, old_inertia_csr)
    assert_allclose(new_inertia, new_inertia_csr)