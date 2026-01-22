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
@pytest.mark.parametrize('tol, max_no_improvement', [(0.0001, None), (0, 10)])
def test_minibatch_declared_convergence(capsys, tol, max_no_improvement):
    X, _, centers = make_blobs(centers=3, random_state=0, return_centers=True)
    km = MiniBatchKMeans(n_clusters=3, init=centers, batch_size=20, tol=tol, random_state=0, max_iter=10, n_init=1, verbose=1, max_no_improvement=max_no_improvement)
    km.fit(X)
    assert 1 < km.n_iter_ < 10
    captured = capsys.readouterr()
    if max_no_improvement is None:
        assert 'Converged (small centers change)' in captured.out
    if tol == 0:
        assert 'Converged (lack of improvement in inertia)' in captured.out