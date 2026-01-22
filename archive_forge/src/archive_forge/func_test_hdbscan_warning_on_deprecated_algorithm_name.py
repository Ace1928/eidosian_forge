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
def test_hdbscan_warning_on_deprecated_algorithm_name():
    msg = "`algorithm='kdtree'`has been deprecated in 1.4 and will be renamed to'kd_tree'`in 1.6. To keep the past behaviour, set `algorithm='kd_tree'`."
    with pytest.warns(FutureWarning, match=msg):
        HDBSCAN(algorithm='kdtree').fit(X)
    msg = "`algorithm='balltree'`has been deprecated in 1.4 and will be renamed to'ball_tree'`in 1.6. To keep the past behaviour, set `algorithm='ball_tree'`."
    with pytest.warns(FutureWarning, match=msg):
        HDBSCAN(algorithm='balltree').fit(X)