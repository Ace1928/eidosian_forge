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
def test_uniform_targets():
    enet = ElasticNetCV(n_alphas=3)
    m_enet = MultiTaskElasticNetCV(n_alphas=3)
    lasso = LassoCV(n_alphas=3)
    m_lasso = MultiTaskLassoCV(n_alphas=3)
    models_single_task = (enet, lasso)
    models_multi_task = (m_enet, m_lasso)
    rng = np.random.RandomState(0)
    X_train = rng.random_sample(size=(10, 3))
    X_test = rng.random_sample(size=(10, 3))
    y1 = np.empty(10)
    y2 = np.empty((10, 2))
    for model in models_single_task:
        for y_values in (0, 5):
            y1.fill(y_values)
            assert_array_equal(model.fit(X_train, y1).predict(X_test), y1)
            assert_array_equal(model.alphas_, [np.finfo(float).resolution] * 3)
    for model in models_multi_task:
        for y_values in (0, 5):
            y2[:, 0].fill(y_values)
            y2[:, 1].fill(2 * y_values)
            assert_array_equal(model.fit(X_train, y2).predict(X_test), y2)
            assert_array_equal(model.alphas_, [np.finfo(float).resolution] * 3)