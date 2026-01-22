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
@pytest.mark.parametrize('gb, dataset_maker, init_estimator', [(GradientBoostingClassifier, make_classification, DummyClassifier), (GradientBoostingClassifier, _make_multiclass, DummyClassifier), (GradientBoostingRegressor, make_regression, DummyRegressor)], ids=['binary classification', 'multiclass classification', 'regression'])
def test_gradient_boosting_with_init(gb, dataset_maker, init_estimator, global_random_seed):
    X, y = dataset_maker()
    sample_weight = np.random.RandomState(global_random_seed).rand(100)
    init_est = init_estimator()
    gb(init=init_est).fit(X, y, sample_weight=sample_weight)
    init_est = NoSampleWeightWrapper(init_estimator())
    gb(init=init_est).fit(X, y)
    with pytest.raises(ValueError, match='estimator.*does not support sample weights'):
        gb(init=init_est).fit(X, y, sample_weight=sample_weight)