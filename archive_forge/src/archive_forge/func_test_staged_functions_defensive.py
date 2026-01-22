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
@pytest.mark.parametrize('Estimator', GRADIENT_BOOSTING_ESTIMATORS)
def test_staged_functions_defensive(Estimator, global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    X = rng.uniform(size=(10, 3))
    y = (4 * X[:, 0]).astype(int) + 1
    estimator = Estimator()
    estimator.fit(X, y)
    for func in ['predict', 'decision_function', 'predict_proba']:
        staged_func = getattr(estimator, 'staged_' + func, None)
        if staged_func is None:
            continue
        with warnings.catch_warnings(record=True):
            staged_result = list(staged_func(X))
        staged_result[1][:] = 0
        assert np.all(staged_result[0] != 0)