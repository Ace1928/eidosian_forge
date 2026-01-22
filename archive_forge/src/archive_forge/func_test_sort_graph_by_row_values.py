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
@pytest.mark.parametrize('function', [sort_graph_by_row_values, _check_precomputed])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sort_graph_by_row_values(function, csr_container):
    X = csr_container(np.abs(np.random.RandomState(42).randn(10, 10)))
    assert not _is_sorted_by_data(X)
    Xt = function(X)
    assert _is_sorted_by_data(Xt)
    mask = np.random.RandomState(42).randint(2, size=(10, 10))
    X = X.toarray()
    X[mask == 1] = 0
    X = csr_container(X)
    assert not _is_sorted_by_data(X)
    Xt = function(X)
    assert _is_sorted_by_data(Xt)