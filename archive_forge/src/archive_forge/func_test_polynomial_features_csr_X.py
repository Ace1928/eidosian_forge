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
@pytest.mark.parametrize(['deg', 'include_bias', 'interaction_only', 'dtype'], [(1, True, False, int), (2, True, False, int), (2, True, False, np.float32), (2, True, False, np.float64), (3, False, False, np.float64), (3, False, True, np.float64)])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_polynomial_features_csr_X(deg, include_bias, interaction_only, dtype, csr_container):
    rng = np.random.RandomState(0)
    X = rng.randint(0, 2, (100, 2))
    X_csr = csr_container(X)
    est = PolynomialFeatures(deg, include_bias=include_bias, interaction_only=interaction_only)
    Xt_csr = est.fit_transform(X_csr.astype(dtype))
    Xt_dense = est.fit_transform(X.astype(dtype, copy=False))
    assert sparse.issparse(Xt_csr) and Xt_csr.format == 'csr'
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.toarray(), Xt_dense)