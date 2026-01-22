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
@pytest.mark.parametrize(('loss', 'value'), [('squared_error', 0.5), ('absolute_error', 0.0), ('huber', 0.5), ('quantile', 0.5)])
def test_non_uniform_weights_toy_edge_case_reg(loss, value):
    X = [[1, 0], [1, 0], [1, 0], [0, 1]]
    y = [0, 0, 1, 0]
    sample_weight = [0, 0, 1, 1]
    gb = GradientBoostingRegressor(learning_rate=1.0, n_estimators=2, loss=loss)
    gb.fit(X, y, sample_weight=sample_weight)
    assert gb.predict([[1, 0]])[0] >= value