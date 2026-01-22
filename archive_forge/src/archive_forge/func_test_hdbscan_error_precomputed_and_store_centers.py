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
@pytest.mark.parametrize('store_centers', ['centroid', 'medoid'])
def test_hdbscan_error_precomputed_and_store_centers(store_centers):
    """Check that we raise an error if the centers are requested together with
    a precomputed input matrix.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27893
    """
    rng = np.random.RandomState(0)
    X = rng.random((100, 2))
    X_dist = euclidean_distances(X)
    err_msg = 'Cannot store centers when using a precomputed distance matrix.'
    with pytest.raises(ValueError, match=err_msg):
        HDBSCAN(metric='precomputed', store_centers=store_centers).fit(X_dist)