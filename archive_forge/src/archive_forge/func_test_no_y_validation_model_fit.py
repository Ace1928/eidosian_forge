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
@pytest.mark.parametrize('y', ('no_validation', 1j, 99.9, np.nan, 3))
def test_no_y_validation_model_fit(y):
    X, clusters = make_blobs(n_features=6)
    sfs = SequentialFeatureSelector(KMeans(), n_features_to_select=3)
    with pytest.raises((TypeError, ValueError)):
        sfs.fit(X, y)