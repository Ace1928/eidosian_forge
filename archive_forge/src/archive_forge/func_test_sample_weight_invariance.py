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
@pytest.mark.parametrize('estimator', [Lasso(alpha=1.0), ElasticNet(alpha=1.0, l1_ratio=0.1)])
def test_sample_weight_invariance(estimator):
    rng = np.random.RandomState(42)
    X, y = make_regression(n_samples=100, n_features=300, effective_rank=10, n_informative=50, random_state=rng)
    sw = rng.uniform(low=0.01, high=2, size=X.shape[0])
    params = dict(tol=1e-12)
    cutoff = X.shape[0] // 3
    sw_with_null = sw.copy()
    sw_with_null[:cutoff] = 0.0
    X_trimmed, y_trimmed = (X[cutoff:, :], y[cutoff:])
    sw_trimmed = sw[cutoff:]
    reg_trimmed = clone(estimator).set_params(**params).fit(X_trimmed, y_trimmed, sample_weight=sw_trimmed)
    reg_null_weighted = clone(estimator).set_params(**params).fit(X, y, sample_weight=sw_with_null)
    assert_allclose(reg_null_weighted.coef_, reg_trimmed.coef_)
    assert_allclose(reg_null_weighted.intercept_, reg_trimmed.intercept_)
    X_dup = np.concatenate([X, X], axis=0)
    y_dup = np.concatenate([y, y], axis=0)
    sw_dup = np.concatenate([sw, sw], axis=0)
    reg_2sw = clone(estimator).set_params(**params).fit(X, y, sample_weight=2 * sw)
    reg_dup = clone(estimator).set_params(**params).fit(X_dup, y_dup, sample_weight=sw_dup)
    assert_allclose(reg_2sw.coef_, reg_dup.coef_)
    assert_allclose(reg_2sw.intercept_, reg_dup.intercept_)