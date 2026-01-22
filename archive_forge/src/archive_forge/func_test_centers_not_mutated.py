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
@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_centers_not_mutated(Estimator, dtype):
    X_new_type = X.astype(dtype, copy=False)
    centers_new_type = centers.astype(dtype, copy=False)
    km = Estimator(init=centers_new_type, n_clusters=n_clusters, n_init=1)
    km.fit(X_new_type)
    assert not np.may_share_memory(km.cluster_centers_, centers_new_type)