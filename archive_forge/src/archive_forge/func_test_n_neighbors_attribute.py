import re
from math import sqrt
import numpy as np
import pytest
from sklearn import metrics, neighbors
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_n_neighbors_attribute():
    X = iris.data
    clf = neighbors.LocalOutlierFactor(n_neighbors=500).fit(X)
    assert clf.n_neighbors_ == X.shape[0] - 1
    clf = neighbors.LocalOutlierFactor(n_neighbors=500)
    msg = 'n_neighbors will be set to (n_samples - 1)'
    with pytest.warns(UserWarning, match=re.escape(msg)):
        clf.fit(X)
    assert clf.n_neighbors_ == X.shape[0] - 1