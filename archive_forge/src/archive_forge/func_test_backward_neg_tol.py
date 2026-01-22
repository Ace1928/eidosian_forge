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
def test_backward_neg_tol():
    """Check that SequentialFeatureSelector works negative tol

    non-regression test for #25525
    """
    X, y = make_regression(n_features=10, random_state=0)
    lr = LinearRegression()
    initial_score = lr.fit(X, y).score(X, y)
    sfs = SequentialFeatureSelector(lr, n_features_to_select='auto', direction='backward', tol=-0.001)
    Xr = sfs.fit_transform(X, y)
    new_score = lr.fit(Xr, y).score(Xr, y)
    assert 0 < sfs.get_support().sum() < X.shape[1]
    assert new_score < initial_score