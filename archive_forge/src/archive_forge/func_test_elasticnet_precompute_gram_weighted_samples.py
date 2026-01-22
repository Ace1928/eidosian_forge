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
def test_elasticnet_precompute_gram_weighted_samples():
    X, y, _, _ = build_dataset()
    rng = np.random.RandomState(0)
    sample_weight = rng.lognormal(size=y.shape)
    w_norm = sample_weight * (y.shape / np.sum(sample_weight))
    X_c = X - np.average(X, axis=0, weights=w_norm)
    X_r = X_c * np.sqrt(w_norm)[:, np.newaxis]
    gram = np.dot(X_r.T, X_r)
    clf1 = ElasticNet(alpha=0.01, precompute=gram)
    clf1.fit(X_c, y, sample_weight=sample_weight)
    clf2 = ElasticNet(alpha=0.01, precompute=False)
    clf2.fit(X, y, sample_weight=sample_weight)
    assert_allclose(clf1.coef_, clf2.coef_)