import pickle
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import datasets, linear_model, metrics
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import _sgd_fast as sgd_fast
from sklearn.linear_model import _stochastic_gradient
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, scale
from sklearn.svm import OneClassSVM
from sklearn.utils._testing import (
@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
def test_elasticnet_convergence(klass):
    n_samples, n_features = (1000, 5)
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    ground_truth_coef = rng.randn(n_features)
    y = np.dot(X, ground_truth_coef)
    for alpha in [0.01, 0.001]:
        for l1_ratio in [0.5, 0.8, 1.0]:
            cd = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
            cd.fit(X, y)
            sgd = klass(penalty='elasticnet', max_iter=50, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
            sgd.fit(X, y)
            err_msg = 'cd and sgd did not converge to comparable results for alpha=%f and l1_ratio=%f' % (alpha, l1_ratio)
            assert_almost_equal(cd.coef_, sgd.coef_, decimal=2, err_msg=err_msg)