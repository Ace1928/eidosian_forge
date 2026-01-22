import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('seed', range(10))
@pytest.mark.parametrize('direction', ('forward', 'backward'))
@pytest.mark.parametrize('n_features_to_select, expected_selected_features', [(2, [0, 2]), (1, [2])])
def test_sanity(seed, direction, n_features_to_select, expected_selected_features):
    rng = np.random.RandomState(seed)
    n_samples = 100
    X = rng.randn(n_samples, 3)
    y = 3 * X[:, 0] - 10 * X[:, 2]
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=n_features_to_select, direction=direction, cv=2)
    sfs.fit(X, y)
    assert_array_equal(sfs.get_support(indices=True), expected_selected_features)