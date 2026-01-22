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
@pytest.mark.parametrize('algorithm', ['lloyd', 'elkan'])
def test_kmeans_convergence(algorithm, global_random_seed):
    rnd = np.random.RandomState(global_random_seed)
    X = rnd.normal(size=(5000, 10))
    max_iter = 300
    km = KMeans(algorithm=algorithm, n_clusters=5, random_state=global_random_seed, n_init=1, tol=0, max_iter=max_iter).fit(X)
    assert km.n_iter_ < max_iter