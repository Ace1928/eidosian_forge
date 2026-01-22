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
def test_sort_graph_by_row_values_warning(csr_container):
    X = csr_container(np.abs(np.random.RandomState(42).randn(10, 10)))
    assert not _is_sorted_by_data(X)
    with pytest.warns(EfficiencyWarning, match='was not sorted by row values'):
        sort_graph_by_row_values(X, copy=True)
    with pytest.warns(EfficiencyWarning, match='was not sorted by row values'):
        sort_graph_by_row_values(X, copy=True, warn_when_not_sorted=True)
    with pytest.warns(EfficiencyWarning, match='was not sorted by row values'):
        _check_precomputed(X)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sort_graph_by_row_values(X, copy=True, warn_when_not_sorted=False)