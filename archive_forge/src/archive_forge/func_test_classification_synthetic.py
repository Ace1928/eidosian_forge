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
@pytest.mark.parametrize('loss', ('log_loss', 'exponential'))
def test_classification_synthetic(loss, global_random_seed):
    X, y = datasets.make_hastie_10_2(n_samples=2000, random_state=global_random_seed)
    split_idx = 500
    X_train, X_test = (X[:split_idx], X[split_idx:])
    y_train, y_test = (y[:split_idx], y[split_idx:])
    common_params = {'max_depth': 1, 'learning_rate': 1.0, 'loss': loss, 'random_state': global_random_seed}
    gbrt_10_stumps = GradientBoostingClassifier(n_estimators=10, **common_params)
    gbrt_10_stumps.fit(X_train, y_train)
    gbrt_50_stumps = GradientBoostingClassifier(n_estimators=50, **common_params)
    gbrt_50_stumps.fit(X_train, y_train)
    assert gbrt_10_stumps.score(X_test, y_test) < gbrt_50_stumps.score(X_test, y_test)
    common_params = {'n_estimators': 200, 'learning_rate': 1.0, 'loss': loss, 'random_state': global_random_seed}
    gbrt_stumps = GradientBoostingClassifier(max_depth=1, **common_params)
    gbrt_stumps.fit(X_train, y_train)
    gbrt_10_nodes = GradientBoostingClassifier(max_leaf_nodes=10, **common_params)
    gbrt_10_nodes.fit(X_train, y_train)
    assert gbrt_stumps.score(X_test, y_test) > gbrt_10_nodes.score(X_test, y_test)