import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble._gb import _safe_divide
from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.svm import NuSVR
from sklearn.utils import check_random_state, tosequence
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_feature_importance_regression(fetch_california_housing_fxt, global_random_seed):
    """Test that Gini importance is calculated correctly.

    This test follows the example from [1]_ (pg. 373).

    .. [1] Friedman, J., Hastie, T., & Tibshirani, R. (2001). The elements
       of statistical learning. New York: Springer series in statistics.
    """
    california = fetch_california_housing_fxt()
    X, y = (california.data, california.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=global_random_seed)
    reg = GradientBoostingRegressor(loss='huber', learning_rate=0.1, max_leaf_nodes=6, n_estimators=100, random_state=global_random_seed)
    reg.fit(X_train, y_train)
    sorted_idx = np.argsort(reg.feature_importances_)[::-1]
    sorted_features = [california.feature_names[s] for s in sorted_idx]
    assert sorted_features[0] == 'MedInc'
    assert set(sorted_features[1:4]) == {'Longitude', 'AveOccup', 'Latitude'}