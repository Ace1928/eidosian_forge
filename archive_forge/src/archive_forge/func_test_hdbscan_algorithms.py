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
@pytest.mark.parametrize('algo', ALGORITHMS)
@pytest.mark.parametrize('metric', _VALID_METRICS)
def test_hdbscan_algorithms(algo, metric):
    """
    Tests that HDBSCAN works with the expected combinations of algorithms and
    metrics, or raises the expected errors.
    """
    labels = HDBSCAN(algorithm=algo).fit_predict(X)
    check_label_quality(labels)
    if algo in ('brute', 'auto'):
        return
    ALGOS_TREES = {'kd_tree': KDTree, 'ball_tree': BallTree}
    metric_params = {'mahalanobis': {'V': np.eye(X.shape[1])}, 'seuclidean': {'V': np.ones(X.shape[1])}, 'minkowski': {'p': 2}, 'wminkowski': {'p': 2, 'w': np.ones(X.shape[1])}}.get(metric, None)
    hdb = HDBSCAN(algorithm=algo, metric=metric, metric_params=metric_params)
    if metric not in ALGOS_TREES[algo].valid_metrics:
        with pytest.raises(ValueError):
            hdb.fit(X)
    elif metric == 'wminkowski':
        with pytest.warns(FutureWarning):
            hdb.fit(X)
    else:
        hdb.fit(X)