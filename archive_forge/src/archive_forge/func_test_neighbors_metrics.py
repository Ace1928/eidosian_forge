import re
import warnings
from itertools import product
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import (
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
from sklearn.neighbors._base import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('metric', sorted(set(neighbors.VALID_METRICS['ball_tree']).intersection(neighbors.VALID_METRICS['brute']) - set(['pyfunc', *BOOL_METRICS])) + DISTANCE_METRIC_OBJS)
def test_neighbors_metrics(global_dtype, metric, n_samples=20, n_features=3, n_query_pts=2, n_neighbors=5):
    metric = _parse_metric(metric, global_dtype)
    algorithms = ['brute', 'ball_tree', 'kd_tree']
    X_train = rng.rand(n_samples, n_features).astype(global_dtype, copy=False)
    X_test = rng.rand(n_query_pts, n_features).astype(global_dtype, copy=False)
    metric_params_list = _generate_test_params_for(metric, n_features)
    for metric_params in metric_params_list:
        exclude_kd_tree = False if isinstance(metric, DistanceMetric) else metric not in neighbors.VALID_METRICS['kd_tree'] or ('minkowski' in metric and 'w' in metric_params)
        results = {}
        p = metric_params.pop('p', 2)
        for algorithm in algorithms:
            if isinstance(metric, DistanceMetric) and global_dtype == np.float32:
                if 'tree' in algorithm:
                    pytest.skip('Neither KDTree nor BallTree support 32-bit distance metric objects.')
            neigh = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric, p=p, metric_params=metric_params)
            if exclude_kd_tree and algorithm == 'kd_tree':
                with pytest.raises(ValueError):
                    neigh.fit(X_train)
                continue
            if metric == 'haversine':
                feature_sl = slice(None, 2)
                X_train = np.ascontiguousarray(X_train[:, feature_sl])
                X_test = np.ascontiguousarray(X_test[:, feature_sl])
            neigh.fit(X_train)
            results[algorithm] = neigh.kneighbors(X_test, return_distance=True)
        brute_dst, brute_idx = results['brute']
        ball_tree_dst, ball_tree_idx = results['ball_tree']
        assert_allclose(brute_dst, ball_tree_dst)
        assert_array_equal(brute_idx, ball_tree_idx)
        if not exclude_kd_tree:
            kd_tree_dst, kd_tree_idx = results['kd_tree']
            assert_allclose(brute_dst, kd_tree_dst)
            assert_array_equal(brute_idx, kd_tree_idx)
            assert_allclose(ball_tree_dst, kd_tree_dst)
            assert_array_equal(ball_tree_idx, kd_tree_idx)