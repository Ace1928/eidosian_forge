import numpy as np
import pytest
from scipy import stats
from scipy.spatial import distance
from sklearn.cluster import HDBSCAN
from sklearn.cluster._hdbscan._tree import (
from sklearn.cluster._hdbscan.hdbscan import _OUTLIER_ENCODING
from sklearn.datasets import make_blobs
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics.pairwise import _VALID_METRICS, euclidean_distances
from sklearn.neighbors import BallTree, KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_hdbscan_distance_matrix():
    """
    Tests that HDBSCAN works with precomputed distance matrices, and throws the
    appropriate errors when needed.
    """
    D = euclidean_distances(X)
    D_original = D.copy()
    labels = HDBSCAN(metric='precomputed', copy=True).fit_predict(D)
    assert_allclose(D, D_original)
    check_label_quality(labels)
    msg = 'The precomputed distance matrix.*has shape'
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(metric='precomputed', copy=True).fit_predict(X)
    msg = 'The precomputed distance matrix.*values'
    D[0, 1] = 10
    D[1, 0] = 1
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(metric='precomputed').fit_predict(D)