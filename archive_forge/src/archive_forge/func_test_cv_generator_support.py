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
def test_cv_generator_support():
    """Check that no exception raised when cv is generator

    non-regression test for #25957
    """
    X, y = make_classification(random_state=0)
    groups = np.zeros_like(y, dtype=int)
    groups[y.size // 2:] = 1
    cv = LeaveOneGroupOut()
    splits = cv.split(X, y, groups=groups)
    knc = KNeighborsClassifier(n_neighbors=5)
    sfs = SequentialFeatureSelector(knc, n_features_to_select=5, cv=splits)
    sfs.fit(X, y)