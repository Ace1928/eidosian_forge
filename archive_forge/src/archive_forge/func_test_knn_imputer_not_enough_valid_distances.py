import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('na', [-1, np.nan])
@pytest.mark.parametrize('weights', ['uniform', 'distance'])
def test_knn_imputer_not_enough_valid_distances(na, weights):
    X1 = np.array([[na, 11], [na, 1], [3, na]])
    X1_imputed = np.array([[3, 11], [3, 1], [3, 6]])
    knn = KNNImputer(missing_values=na, n_neighbors=1, weights=weights)
    assert_allclose(knn.fit_transform(X1), X1_imputed)
    X2 = np.array([[4, na]])
    X2_imputed = np.array([[4, 6]])
    assert_allclose(knn.transform(X2), X2_imputed)