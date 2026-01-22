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
def test_hdbscan_too_many_min_samples():
    """
    Tests that HDBSCAN correctly raises an error when setting `min_samples`
    larger than the number of samples.
    """
    hdb = HDBSCAN(min_samples=len(X) + 1)
    msg = 'min_samples (.*) must be at most'
    with pytest.raises(ValueError, match=msg):
        hdb.fit(X)