import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('na', [np.nan, -1])
def test_knn_imputer_all_samples_are_neighbors(na):
    X = np.array([[0, 0], [na, 2], [4, 3], [5, na], [7, 7], [na, 8], [14, 13]])
    X_imputed = np.array([[0, 0], [6, 2], [4, 3], [5, 5.5], [7, 7], [6, 8], [14, 13]])
    n_neighbors = X.shape[0] - 1
    imputer = KNNImputer(n_neighbors=n_neighbors, missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed)
    n_neighbors = X.shape[0]
    imputer_plus1 = KNNImputer(n_neighbors=n_neighbors, missing_values=na)
    assert_allclose(imputer_plus1.fit_transform(X), X_imputed)