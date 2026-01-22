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
def test_zero_estimator_reg(global_random_seed):
    baseline = DummyRegressor(strategy='mean').fit(X_reg, y_reg)
    mse_baseline = mean_squared_error(baseline.predict(X_reg), y_reg)
    est = GradientBoostingRegressor(n_estimators=5, max_depth=1, random_state=global_random_seed, init='zero', learning_rate=0.5)
    est.fit(X_reg, y_reg)
    y_pred = est.predict(X_reg)
    mse_gbdt = mean_squared_error(y_reg, y_pred)
    assert mse_gbdt < mse_baseline