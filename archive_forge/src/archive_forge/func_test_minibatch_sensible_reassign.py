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
def test_minibatch_sensible_reassign(global_random_seed):
    zeroed_X, true_labels = make_blobs(n_samples=100, centers=5, random_state=global_random_seed)
    zeroed_X[::2, :] = 0
    km = MiniBatchKMeans(n_clusters=20, batch_size=10, random_state=global_random_seed, init='random').fit(zeroed_X)
    assert km.cluster_centers_.any(axis=1).sum() > 10
    km = MiniBatchKMeans(n_clusters=20, batch_size=200, random_state=global_random_seed, init='random').fit(zeroed_X)
    assert km.cluster_centers_.any(axis=1).sum() > 10
    km = MiniBatchKMeans(n_clusters=20, random_state=global_random_seed, init='random')
    for i in range(100):
        km.partial_fit(zeroed_X)
    assert km.cluster_centers_.any(axis=1).sum() > 10