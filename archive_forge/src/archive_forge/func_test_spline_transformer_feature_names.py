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
def test_spline_transformer_feature_names():
    """Test that SplineTransformer generates correct features name."""
    X = np.arange(20).reshape(10, 2)
    splt = SplineTransformer(n_knots=3, degree=3, include_bias=True).fit(X)
    feature_names = splt.get_feature_names_out()
    assert_array_equal(feature_names, ['x0_sp_0', 'x0_sp_1', 'x0_sp_2', 'x0_sp_3', 'x0_sp_4', 'x1_sp_0', 'x1_sp_1', 'x1_sp_2', 'x1_sp_3', 'x1_sp_4'])
    splt = SplineTransformer(n_knots=3, degree=3, include_bias=False).fit(X)
    feature_names = splt.get_feature_names_out(['a', 'b'])
    assert_array_equal(feature_names, ['a_sp_0', 'a_sp_1', 'a_sp_2', 'a_sp_3', 'b_sp_0', 'b_sp_1', 'b_sp_2', 'b_sp_3'])