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
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('l1_ratio', [0, 0.5, 1])
@pytest.mark.parametrize('precompute', [False, True])
@pytest.mark.parametrize('sparse_container', [None] + CSC_CONTAINERS)
def test_enet_cv_sample_weight_consistency(fit_intercept, l1_ratio, precompute, sparse_container):
    """Test that the impact of sample_weight is consistent."""
    rng = np.random.RandomState(0)
    n_samples, n_features = (10, 5)
    X = rng.rand(n_samples, n_features)
    y = X.sum(axis=1) + rng.rand(n_samples)
    params = dict(l1_ratio=l1_ratio, fit_intercept=fit_intercept, precompute=precompute, tol=1e-06, cv=3)
    if sparse_container is not None:
        X = sparse_container(X)
    if l1_ratio == 0:
        params.pop('l1_ratio', None)
        reg = LassoCV(**params).fit(X, y)
    else:
        reg = ElasticNetCV(**params).fit(X, y)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_
    sample_weight = np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-06)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)
    sample_weight = 123.0
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-06)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)
    sample_weight = 2 * np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-06)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)