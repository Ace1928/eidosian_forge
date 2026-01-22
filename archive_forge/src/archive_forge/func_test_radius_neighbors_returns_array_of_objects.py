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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_radius_neighbors_returns_array_of_objects(csr_container):
    X = csr_container(np.ones((4, 4)))
    X.setdiag([0, 0, 0, 0])
    nbrs = neighbors.NearestNeighbors(radius=0.5, algorithm='auto', leaf_size=30, metric='precomputed').fit(X)
    neigh_dist, neigh_ind = nbrs.radius_neighbors(X, return_distance=True)
    expected_dist = np.empty(X.shape[0], dtype=object)
    expected_dist[:] = [np.array([0]), np.array([0]), np.array([0]), np.array([0])]
    expected_ind = np.empty(X.shape[0], dtype=object)
    expected_ind[:] = [np.array([0]), np.array([1]), np.array([2]), np.array([3])]
    assert_array_equal(neigh_dist, expected_dist)
    assert_array_equal(neigh_ind, expected_ind)