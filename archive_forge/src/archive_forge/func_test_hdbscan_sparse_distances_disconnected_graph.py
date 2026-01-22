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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_hdbscan_sparse_distances_disconnected_graph(csr_container):
    """
    Tests that HDBSCAN raises the correct error when the distance matrix
    has multiple connected components.
    """
    X = np.zeros((20, 20))
    X[:5, :5] = 1
    X[5:, 15:] = 1
    X = X + X.T
    X = csr_container(X)
    msg = 'HDBSCAN cannot be perfomed on a disconnected graph'
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(metric='precomputed').fit(X)