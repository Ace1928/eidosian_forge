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
@pytest.mark.parametrize('kwargs', ({'init': np.str_('k-means++')}, {'init': [[0, 0], [1, 1]], 'n_init': 1}))
def test_kmeans_with_array_like_or_np_scalar_init(kwargs):
    """Check that init works with numpy scalar strings.

    Non-regression test for #21964.
    """
    X = np.asarray([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=np.float64)
    clustering = KMeans(n_clusters=2, **kwargs)
    clustering.fit(X)