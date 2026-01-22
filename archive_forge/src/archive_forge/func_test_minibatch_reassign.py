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
@pytest.mark.parametrize('input_data', [X] + X_as_any_csr, ids=data_containers_ids)
def test_minibatch_reassign(input_data, global_random_seed):
    perfect_centers = np.empty((n_clusters, n_features))
    for i in range(n_clusters):
        perfect_centers[i] = X[true_labels == i].mean(axis=0)
    sample_weight = np.ones(n_samples)
    centers_new = np.empty_like(perfect_centers)
    score_before = -_labels_inertia(input_data, sample_weight, perfect_centers, 1)[1]
    _mini_batch_step(input_data, sample_weight, perfect_centers, centers_new, np.zeros(n_clusters), np.random.RandomState(global_random_seed), random_reassign=True, reassignment_ratio=1)
    score_after = -_labels_inertia(input_data, sample_weight, centers_new, 1)[1]
    assert score_before > score_after
    _mini_batch_step(input_data, sample_weight, perfect_centers, centers_new, np.zeros(n_clusters), np.random.RandomState(global_random_seed), random_reassign=True, reassignment_ratio=1e-15)
    assert_allclose(centers_new, perfect_centers)