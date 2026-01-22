import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('working_memory', [None, 0])
@pytest.mark.parametrize('na', [-1, np.nan])
@pytest.mark.filterwarnings('ignore:adhere to working_memory')
def test_knn_imputer_with_simple_example(na, working_memory):
    X = np.array([[0, na, 0, na], [1, 1, 1, na], [2, 2, na, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6], [na, 7, 7, 7]])
    r0c1 = np.mean(X[1:6, 1])
    r0c3 = np.mean(X[2:-1, -1])
    r1c3 = np.mean(X[2:-1, -1])
    r2c2 = np.mean(X[[0, 1, 3, 4, 5], 2])
    r7c0 = np.mean(X[2:-1, 0])
    X_imputed = np.array([[0, r0c1, 0, r0c3], [1, 1, 1, r1c3], [2, 2, r2c2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6], [r7c0, 7, 7, 7]])
    with config_context(working_memory=working_memory):
        imputer_comp = KNNImputer(missing_values=na)
        assert_allclose(imputer_comp.fit_transform(X), X_imputed)