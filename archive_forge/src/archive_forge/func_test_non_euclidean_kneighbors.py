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
def test_non_euclidean_kneighbors():
    rng = np.random.RandomState(0)
    X = rng.rand(5, 5)
    dist_array = pairwise_distances(X).flatten()
    np.sort(dist_array)
    radius = dist_array[15]
    for metric in ['manhattan', 'chebyshev']:
        nbrs_graph = neighbors.kneighbors_graph(X, 3, metric=metric, mode='connectivity', include_self=True).toarray()
        nbrs1 = neighbors.NearestNeighbors(n_neighbors=3, metric=metric).fit(X)
        assert_array_equal(nbrs_graph, nbrs1.kneighbors_graph(X).toarray())
    for metric in ['manhattan', 'chebyshev']:
        nbrs_graph = neighbors.radius_neighbors_graph(X, radius, metric=metric, mode='connectivity', include_self=True).toarray()
        nbrs1 = neighbors.NearestNeighbors(metric=metric, radius=radius).fit(X)
        assert_array_equal(nbrs_graph, nbrs1.radius_neighbors_graph(X).toarray())
    X_nbrs = neighbors.NearestNeighbors(n_neighbors=3, metric='manhattan')
    X_nbrs.fit(X)
    with pytest.raises(ValueError):
        neighbors.kneighbors_graph(X_nbrs, 3, metric='euclidean')
    X_nbrs = neighbors.NearestNeighbors(radius=radius, metric='manhattan')
    X_nbrs.fit(X)
    with pytest.raises(ValueError):
        neighbors.radius_neighbors_graph(X_nbrs, radius, metric='euclidean')