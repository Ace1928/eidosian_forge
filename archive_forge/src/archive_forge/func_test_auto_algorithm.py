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
@pytest.mark.parametrize('X, metric, metric_params, expected_algo', [(np.random.randint(10, size=(10, 10)), 'precomputed', None, 'brute'), (np.random.randn(10, 20), 'euclidean', None, 'brute'), (np.random.randn(8, 5), 'euclidean', None, 'brute'), (np.random.randn(10, 5), 'euclidean', None, 'kd_tree'), (np.random.randn(10, 5), 'seuclidean', {'V': [2] * 5}, 'ball_tree'), (np.random.randn(10, 5), 'correlation', None, 'brute')])
def test_auto_algorithm(X, metric, metric_params, expected_algo):
    model = neighbors.NearestNeighbors(n_neighbors=4, algorithm='auto', metric=metric, metric_params=metric_params)
    model.fit(X)
    assert model._fit_method == expected_algo