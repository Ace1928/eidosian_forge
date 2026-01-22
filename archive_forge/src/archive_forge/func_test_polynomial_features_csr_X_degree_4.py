import sys
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse import random as sparse_random
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._csr_polynomial_expansion import (
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize(['include_bias', 'interaction_only'], [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_polynomial_features_csr_X_degree_4(include_bias, interaction_only, csr_container):
    X_csr = csr_container(sparse_random(1000, 10, 0.5, random_state=0))
    X = X_csr.toarray()
    est = PolynomialFeatures(4, include_bias=include_bias, interaction_only=interaction_only)
    Xt_csr = est.fit_transform(X_csr)
    Xt_dense = est.fit_transform(X)
    assert sparse.issparse(Xt_csr) and Xt_csr.format == 'csr'
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.toarray(), Xt_dense)