import numpy as np
import pytest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_mutual_info_classif_mixed(global_dtype):
    rng = check_random_state(0)
    X = rng.rand(1000, 3).astype(global_dtype, copy=False)
    X[:, 1] += X[:, 0]
    y = (0.5 * X[:, 0] + X[:, 2] > 0.5).astype(int)
    X[:, 2] = X[:, 2] > 0.5
    mi = mutual_info_classif(X, y, discrete_features=[2], n_neighbors=3, random_state=0)
    assert_array_equal(np.argsort(-mi), [2, 0, 1])
    for n_neighbors in [5, 7, 9]:
        mi_nn = mutual_info_classif(X, y, discrete_features=[2], n_neighbors=n_neighbors, random_state=0)
        assert mi_nn[0] > mi[0]
        assert mi_nn[1] > mi[1]
        assert mi_nn[2] == mi[2]