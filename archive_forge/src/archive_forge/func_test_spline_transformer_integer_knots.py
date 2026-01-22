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
@pytest.mark.parametrize('extrapolation', ['continue', 'periodic'])
def test_spline_transformer_integer_knots(extrapolation):
    """Test that SplineTransformer accepts integer value knot positions."""
    X = np.arange(20).reshape(10, 2)
    knots = [[0, 1], [1, 2], [5, 5], [11, 10], [12, 11]]
    _ = SplineTransformer(degree=3, knots=knots, extrapolation=extrapolation).fit_transform(X)