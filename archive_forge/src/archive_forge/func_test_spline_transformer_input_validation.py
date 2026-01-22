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
@pytest.mark.parametrize('params, err_msg', [({'knots': [[1]]}, 'Number of knots, knots.shape\\[0\\], must be >= 2.'), ({'knots': [[1, 1], [2, 2]]}, 'knots.shape\\[1\\] == n_features is violated'), ({'knots': [[1], [0]]}, 'knots must be sorted without duplicates.')])
def test_spline_transformer_input_validation(params, err_msg):
    """Test that we raise errors for invalid input in SplineTransformer."""
    X = [[1], [2]]
    with pytest.raises(ValueError, match=err_msg):
        SplineTransformer(**params).fit(X)