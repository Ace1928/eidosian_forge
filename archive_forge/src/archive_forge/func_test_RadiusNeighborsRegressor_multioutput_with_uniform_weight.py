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
def test_RadiusNeighborsRegressor_multioutput_with_uniform_weight():
    rng = check_random_state(0)
    n_features = 5
    n_samples = 40
    n_output = 4
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples, n_output)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    for algorithm, weights in product(ALGORITHMS, [None, 'uniform']):
        rnn = neighbors.RadiusNeighborsRegressor(weights=weights, algorithm=algorithm)
        rnn.fit(X_train, y_train)
        neigh_idx = rnn.radius_neighbors(X_test, return_distance=False)
        y_pred_idx = np.array([np.mean(y_train[idx], axis=0) for idx in neigh_idx])
        y_pred_idx = np.array(y_pred_idx)
        y_pred = rnn.predict(X_test)
        assert y_pred_idx.shape == y_test.shape
        assert y_pred.shape == y_test.shape
        assert_allclose(y_pred, y_pred_idx)