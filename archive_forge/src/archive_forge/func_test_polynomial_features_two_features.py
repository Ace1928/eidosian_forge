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
@pytest.mark.parametrize('degree, include_bias, interaction_only, indices', [(2, True, False, slice(0, 6)), (2, False, False, slice(1, 6)), (2, True, True, [0, 1, 2, 4]), (2, False, True, [1, 2, 4]), ((2, 2), True, False, [0, 3, 4, 5]), ((2, 2), False, False, [3, 4, 5]), ((2, 2), True, True, [0, 4]), ((2, 2), False, True, [4]), (3, True, False, slice(None, None)), (3, False, False, slice(1, None)), (3, True, True, [0, 1, 2, 4]), (3, False, True, [1, 2, 4]), ((2, 3), True, False, [0, 3, 4, 5, 6, 7, 8, 9]), ((2, 3), False, False, slice(3, None)), ((2, 3), True, True, [0, 4]), ((2, 3), False, True, [4]), ((3, 3), True, False, [0, 6, 7, 8, 9]), ((3, 3), False, False, [6, 7, 8, 9]), ((3, 3), True, True, [0]), ((3, 3), False, True, [])])
@pytest.mark.parametrize('X_container', [None] + CSR_CONTAINERS + CSC_CONTAINERS)
def test_polynomial_features_two_features(two_features_degree3, degree, include_bias, interaction_only, indices, X_container):
    """Test PolynomialFeatures on 2 features up to degree 3."""
    X, P = two_features_degree3
    if X_container is not None:
        X = X_container(X)
    tf = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=interaction_only).fit(X)
    out = tf.transform(X)
    if X_container is not None:
        out = out.toarray()
    assert_allclose(out, P[:, indices])
    if tf.n_output_features_ > 0:
        assert tf.powers_.shape == (tf.n_output_features_, tf.n_features_in_)