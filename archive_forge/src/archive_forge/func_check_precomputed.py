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
@pytest.mark.filterwarnings('ignore:EfficiencyWarning')
def check_precomputed(make_train_test, estimators):
    """Tests unsupervised NearestNeighbors with a distance matrix."""
    rng = np.random.RandomState(42)
    X = rng.random_sample((10, 4))
    Y = rng.random_sample((3, 4))
    DXX, DYX = make_train_test(X, Y)
    for method in ['kneighbors']:
        nbrs_X = neighbors.NearestNeighbors(n_neighbors=3)
        nbrs_X.fit(X)
        dist_X, ind_X = getattr(nbrs_X, method)(Y)
        nbrs_D = neighbors.NearestNeighbors(n_neighbors=3, algorithm='brute', metric='precomputed')
        nbrs_D.fit(DXX)
        dist_D, ind_D = getattr(nbrs_D, method)(DYX)
        assert_allclose(dist_X, dist_D)
        assert_array_equal(ind_X, ind_D)
        nbrs_D = neighbors.NearestNeighbors(n_neighbors=3, algorithm='auto', metric='precomputed')
        nbrs_D.fit(DXX)
        dist_D, ind_D = getattr(nbrs_D, method)(DYX)
        assert_allclose(dist_X, dist_D)
        assert_array_equal(ind_X, ind_D)
        dist_X, ind_X = getattr(nbrs_X, method)(None)
        dist_D, ind_D = getattr(nbrs_D, method)(None)
        assert_allclose(dist_X, dist_D)
        assert_array_equal(ind_X, ind_D)
        with pytest.raises(ValueError):
            getattr(nbrs_D, method)(X)
    target = np.arange(X.shape[0])
    for Est in estimators:
        est = Est(metric='euclidean')
        est.radius = est.n_neighbors = 1
        pred_X = est.fit(X, target).predict(Y)
        est.metric = 'precomputed'
        pred_D = est.fit(DXX, target).predict(DYX)
        assert_allclose(pred_X, pred_D)