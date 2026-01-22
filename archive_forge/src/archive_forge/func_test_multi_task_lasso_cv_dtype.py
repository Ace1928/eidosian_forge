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
def test_multi_task_lasso_cv_dtype():
    n_samples, n_features = (10, 3)
    rng = np.random.RandomState(42)
    X = rng.binomial(1, 0.5, size=(n_samples, n_features))
    X = X.astype(int)
    y = X[:, [0, 0]].copy()
    est = MultiTaskLassoCV(n_alphas=5, fit_intercept=True).fit(X, y)
    assert_array_almost_equal(est.coef_, [[1, 0, 0]] * 2, decimal=3)