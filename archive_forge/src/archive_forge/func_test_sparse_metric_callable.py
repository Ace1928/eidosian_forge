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
def test_sparse_metric_callable(csr_container):

    def sparse_metric(x, y):
        assert issparse(x) and issparse(y)
        return x.dot(y.T).toarray().item()
    X = csr_container([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 0, 1, 0, 0]])
    Y = csr_container([[1, 1, 0, 1, 1], [1, 0, 0, 1, 1]])
    nn = neighbors.NearestNeighbors(algorithm='brute', n_neighbors=2, metric=sparse_metric).fit(X)
    N = nn.kneighbors(Y, return_distance=False)
    gold_standard_nn = np.array([[2, 1], [2, 1]])
    assert_array_equal(N, gold_standard_nn)