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
def test_max_features():
    X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
    _, n_features = X.shape
    X_train = X[:2000]
    y_train = y[:2000]
    gbrt = GradientBoostingClassifier(n_estimators=1, max_features=None)
    gbrt.fit(X_train, y_train)
    assert gbrt.max_features_ == n_features
    gbrt = GradientBoostingRegressor(n_estimators=1, max_features=None)
    gbrt.fit(X_train, y_train)
    assert gbrt.max_features_ == n_features
    gbrt = GradientBoostingRegressor(n_estimators=1, max_features=0.3)
    gbrt.fit(X_train, y_train)
    assert gbrt.max_features_ == int(n_features * 0.3)
    gbrt = GradientBoostingRegressor(n_estimators=1, max_features='sqrt')
    gbrt.fit(X_train, y_train)
    assert gbrt.max_features_ == int(np.sqrt(n_features))
    gbrt = GradientBoostingRegressor(n_estimators=1, max_features='log2')
    gbrt.fit(X_train, y_train)
    assert gbrt.max_features_ == int(np.log2(n_features))
    gbrt = GradientBoostingRegressor(n_estimators=1, max_features=0.01 / X.shape[1])
    gbrt.fit(X_train, y_train)
    assert gbrt.max_features_ == 1