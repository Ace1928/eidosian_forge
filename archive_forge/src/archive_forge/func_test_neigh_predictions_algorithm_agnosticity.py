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
@pytest.mark.parametrize('n_samples, n_features, n_query_pts', [(100, 100, 10), (1000, 5, 100)])
@pytest.mark.parametrize('metric', COMMON_VALID_METRICS + DISTANCE_METRIC_OBJS)
@pytest.mark.parametrize('n_neighbors, radius', [(1, 100), (50, 500), (100, 1000)])
@pytest.mark.parametrize('NeighborsMixinSubclass', [neighbors.KNeighborsClassifier, neighbors.KNeighborsRegressor, neighbors.RadiusNeighborsClassifier, neighbors.RadiusNeighborsRegressor])
def test_neigh_predictions_algorithm_agnosticity(global_dtype, n_samples, n_features, n_query_pts, metric, n_neighbors, radius, NeighborsMixinSubclass):
    metric = _parse_metric(metric, global_dtype)
    if isinstance(metric, DistanceMetric):
        if 'Classifier' in NeighborsMixinSubclass.__name__:
            pytest.skip('Metrics of type `DistanceMetric` are not yet supported for classifiers.')
        if 'Radius' in NeighborsMixinSubclass.__name__:
            pytest.skip('Metrics of type `DistanceMetric` are not yet supported for radius-neighbor estimators.')
    local_rng = np.random.RandomState(0)
    X = local_rng.rand(n_samples, n_features).astype(global_dtype, copy=False)
    y = local_rng.randint(3, size=n_samples)
    query = local_rng.rand(n_query_pts, n_features).astype(global_dtype, copy=False)
    predict_results = []
    parameter = n_neighbors if issubclass(NeighborsMixinSubclass, KNeighborsMixin) else radius
    for algorithm in ALGORITHMS:
        if isinstance(metric, DistanceMetric) and global_dtype == np.float32:
            if 'tree' in algorithm:
                pytest.skip('Neither KDTree nor BallTree support 32-bit distance metric objects.')
        neigh = NeighborsMixinSubclass(parameter, algorithm=algorithm, metric=metric)
        neigh.fit(X, y)
        predict_results.append(neigh.predict(query))
    for i in range(len(predict_results) - 1):
        algorithm = ALGORITHMS[i]
        next_algorithm = ALGORITHMS[i + 1]
        predictions, next_predictions = (predict_results[i], predict_results[i + 1])
        assert_allclose(predictions, next_predictions, err_msg=f"The '{algorithm}' and '{next_algorithm}' algorithms return different predictions.")