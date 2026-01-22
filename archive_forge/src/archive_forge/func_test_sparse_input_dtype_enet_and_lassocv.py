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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sparse_input_dtype_enet_and_lassocv(csr_container):
    X, y, _, _ = build_dataset(n_features=10)
    clf = ElasticNetCV(n_alphas=5)
    clf.fit(csr_container(X), y)
    clf1 = ElasticNetCV(n_alphas=5)
    clf1.fit(csr_container(X, dtype=np.float32), y)
    assert_almost_equal(clf.alpha_, clf1.alpha_, decimal=6)
    assert_almost_equal(clf.coef_, clf1.coef_, decimal=6)
    clf = LassoCV(n_alphas=5)
    clf.fit(csr_container(X), y)
    clf1 = LassoCV(n_alphas=5)
    clf1.fit(csr_container(X, dtype=np.float32), y)
    assert_almost_equal(clf.alpha_, clf1.alpha_, decimal=6)
    assert_almost_equal(clf.coef_, clf1.coef_, decimal=6)