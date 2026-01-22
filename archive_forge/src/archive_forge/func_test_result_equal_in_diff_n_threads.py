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
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_result_equal_in_diff_n_threads(Estimator, global_random_seed):
    rnd = np.random.RandomState(global_random_seed)
    X = rnd.normal(size=(50, 10))
    with threadpool_limits(limits=1, user_api='openmp'):
        result_1 = Estimator(n_clusters=n_clusters, random_state=global_random_seed).fit(X).labels_
    with threadpool_limits(limits=2, user_api='openmp'):
        result_2 = Estimator(n_clusters=n_clusters, random_state=global_random_seed).fit(X).labels_
    assert_array_equal(result_1, result_2)