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
@skip_if_32bit
def test_huber_exact_backward_compat():
    """Test huber GBT backward compat on a simple dataset.

    The results to compare against are taken from scikit-learn v1.2.0.
    """
    n_samples = 10
    y = np.arange(n_samples)
    x1 = np.minimum(y, n_samples / 2)
    x2 = np.minimum(-y, -n_samples / 2)
    X = np.c_[x1, x2]
    gbt = GradientBoostingRegressor(loss='huber', n_estimators=100, alpha=0.8).fit(X, y)
    assert_allclose(gbt._loss.closs.delta, 0.0001655688041282133)
    pred_result = np.array([0.000148120765, 0.999949174, 2.00116957, 2.99986716, 4.00012064, 5.00002462, 5.99998898, 6.99692549, 8.00006356, 8.99985099])
    assert_allclose(gbt.predict(X), pred_result, rtol=1e-08)
    train_score = np.array([2.59484709e-07, 2.191659e-07, 1.89644782e-07, 1.64556454e-07, 1.3870511e-07, 1.20373736e-07, 1.04746082e-07, 9.13835687e-08, 8.20245756e-08, 7.17122188e-08])
    assert_allclose(gbt.train_score_[-10:], train_score, rtol=1e-08)