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
def test_huber_vs_mean_and_median():
    """Check that huber lies between absolute and squared error."""
    n_rep = 100
    n_samples = 10
    y = np.tile(np.arange(n_samples), n_rep)
    x1 = np.minimum(y, n_samples / 2)
    x2 = np.minimum(-y, -n_samples / 2)
    X = np.c_[x1, x2]
    rng = np.random.RandomState(42)
    y = y + rng.exponential(scale=1, size=y.shape)
    gbt_absolute_error = GradientBoostingRegressor(loss='absolute_error').fit(X, y)
    gbt_huber = GradientBoostingRegressor(loss='huber').fit(X, y)
    gbt_squared_error = GradientBoostingRegressor().fit(X, y)
    gbt_huber_predictions = gbt_huber.predict(X)
    assert np.all(gbt_absolute_error.predict(X) <= gbt_huber_predictions)
    assert np.all(gbt_huber_predictions <= gbt_squared_error.predict(X))