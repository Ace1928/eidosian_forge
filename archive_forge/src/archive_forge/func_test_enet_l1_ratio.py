import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_enet_l1_ratio():
    msg = 'Automatic alpha grid generation is not supported for l1_ratio=0. Please supply a grid by providing your estimator with the appropriate `alphas=` argument.'
    X = np.array([[1, 2, 4, 5, 8], [3, 5, 7, 7, 8]]).T
    y = np.array([12, 10, 11, 21, 5])
    with pytest.raises(ValueError, match=msg):
        ElasticNetCV(l1_ratio=0, random_state=42).fit(X, y)
    with pytest.raises(ValueError, match=msg):
        MultiTaskElasticNetCV(l1_ratio=0, random_state=42).fit(X, y[:, None])
    warning_message = 'Coordinate descent without L1 regularization may lead to unexpected results and is discouraged. Set l1_ratio > 0 to add L1 regularization.'
    est = ElasticNetCV(l1_ratio=[0], alphas=[1])
    with pytest.warns(UserWarning, match=warning_message):
        est.fit(X, y)
    alphas = [0.1, 10]
    estkwds = {'alphas': alphas, 'random_state': 42}
    est_desired = ElasticNetCV(l1_ratio=1e-05, **estkwds)
    est = ElasticNetCV(l1_ratio=0, **estkwds)
    with ignore_warnings():
        est_desired.fit(X, y)
        est.fit(X, y)
    assert_array_almost_equal(est.coef_, est_desired.coef_, decimal=5)
    est_desired = MultiTaskElasticNetCV(l1_ratio=1e-05, **estkwds)
    est = MultiTaskElasticNetCV(l1_ratio=0, **estkwds)
    with ignore_warnings():
        est.fit(X, y[:, None])
        est_desired.fit(X, y[:, None])
    assert_array_almost_equal(est.coef_, est_desired.coef_, decimal=5)