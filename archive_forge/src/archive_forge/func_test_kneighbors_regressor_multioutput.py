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
def test_kneighbors_regressor_multioutput(n_samples=40, n_features=5, n_test_pts=10, n_neighbors=3, random_state=0):
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = np.sqrt((X ** 2).sum(1))
    y /= y.max()
    y = np.vstack([y, y]).T
    y_target = y[:n_test_pts]
    weights = ['uniform', 'distance', _weight_func]
    for algorithm, weights in product(ALGORITHMS, weights):
        knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        knn.fit(X, y)
        epsilon = 1e-05 * (2 * rng.rand(1, n_features) - 1)
        y_pred = knn.predict(X[:n_test_pts] + epsilon)
        assert y_pred.shape == y_target.shape
        assert np.all(np.abs(y_pred - y_target) < 0.3)