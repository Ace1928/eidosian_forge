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
def test_kmeans_warns_less_centers_than_unique_points(global_random_seed):
    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 0]])
    km = KMeans(n_clusters=4, random_state=global_random_seed)
    msg = 'Number of distinct clusters \\(3\\) found smaller than n_clusters \\(4\\). Possibly due to duplicate points in X.'
    with pytest.warns(ConvergenceWarning, match=msg):
        km.fit(X)
        assert set(km.labels_) == set(range(3))