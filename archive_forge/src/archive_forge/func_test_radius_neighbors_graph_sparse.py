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
@pytest.mark.parametrize('n_neighbors', [1, 2, 3])
@pytest.mark.parametrize('mode', ['connectivity', 'distance'])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_radius_neighbors_graph_sparse(n_neighbors, mode, csr_container, seed=36):
    rng = np.random.RandomState(seed)
    X = rng.randn(10, 10)
    Xcsr = csr_container(X)
    assert_allclose(neighbors.radius_neighbors_graph(X, n_neighbors, mode=mode).toarray(), neighbors.radius_neighbors_graph(Xcsr, n_neighbors, mode=mode).toarray())