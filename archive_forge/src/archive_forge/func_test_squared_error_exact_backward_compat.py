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
def test_squared_error_exact_backward_compat():
    """Test squared error GBT backward compat on a simple dataset.

    The results to compare against are taken from scikit-learn v1.2.0.
    """
    n_samples = 10
    y = np.arange(n_samples)
    x1 = np.minimum(y, n_samples / 2)
    x2 = np.minimum(-y, -n_samples / 2)
    X = np.c_[x1, x2]
    gbt = GradientBoostingRegressor(loss='squared_error', n_estimators=100).fit(X, y)
    pred_result = np.array([0.000139245726, 1.00010468, 2.00007043, 3.00004051, 4.00000802, 4.99998972, 5.99996312, 6.99993395, 7.99989372, 8.9998566])
    assert_allclose(gbt.predict(X), pred_result, rtol=1e-08)
    train_score = np.array([4.8724639e-08, 3.95590036e-08, 3.21267865e-08, 2.609703e-08, 2.11820178e-08, 1.71995782e-08, 1.39695549e-08, 1.1339177e-08, 9.19931587e-09, 7.47000575e-09])
    assert_allclose(gbt.train_score_[-10:], train_score, rtol=1e-08)
    sample_weights = np.tile([1, 10], n_samples // 2)
    gbt = GradientBoostingRegressor(loss='squared_error', n_estimators=100).fit(X, y, sample_weight=sample_weights)
    pred_result = np.array([0.000152391462, 1.00011168, 2.00007724, 3.00004638, 4.00001302, 4.99999873, 5.99997093, 6.99994329, 7.9999129, 8.99988727])
    assert_allclose(gbt.predict(X), pred_result, rtol=1e-06, atol=1e-05)
    train_score = np.array([4.12445296e-08, 3.34418322e-08, 2.71151383e-08, 2.19782469e-08, 1.78173649e-08, 1.44461976e-08, 1.17120123e-08, 9.49485678e-09, 7.69772505e-09, 6.24155316e-09])
    assert_allclose(gbt.train_score_[-10:], train_score, rtol=0.001, atol=1e-11)