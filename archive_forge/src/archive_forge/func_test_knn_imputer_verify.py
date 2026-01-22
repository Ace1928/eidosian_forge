import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('na', [np.nan, -1])
def test_knn_imputer_verify(na):
    X = np.array([[1, 0, 0, 1], [2, 1, 2, na], [3, 2, 3, na], [na, 4, 5, 5], [6, na, 6, 7], [8, 8, 8, 8], [16, 15, 18, 19]])
    X_imputed = np.array([[1, 0, 0, 1], [2, 1, 2, 8], [3, 2, 3, 8], [4, 4, 5, 5], [6, 3, 6, 7], [8, 8, 8, 8], [16, 15, 18, 19]])
    imputer = KNNImputer(missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed)
    X = np.array([[1, 0, 0, na], [2, 1, 2, na], [3, 2, 3, na], [4, 4, 5, na], [6, 7, 6, na], [8, 8, 8, na], [20, 20, 20, 20], [22, 22, 22, 22]])
    X_impute_value = (20 + 22) / 2
    X_imputed = np.array([[1, 0, 0, X_impute_value], [2, 1, 2, X_impute_value], [3, 2, 3, X_impute_value], [4, 4, 5, X_impute_value], [6, 7, 6, X_impute_value], [8, 8, 8, X_impute_value], [20, 20, 20, 20], [22, 22, 22, 22]])
    imputer = KNNImputer(missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed)
    X = np.array([[0, 0], [na, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 16]])
    X1 = np.array([[1, 0], [3, 2], [4, na]])
    X_2_1 = (0 + 3 + 6 + 7 + 8) / 5
    X1_imputed = np.array([[1, 0], [3, 2], [4, X_2_1]])
    imputer = KNNImputer(missing_values=na)
    assert_allclose(imputer.fit(X).transform(X1), X1_imputed)