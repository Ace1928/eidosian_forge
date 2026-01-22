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
@pytest.mark.parametrize('params, err_msg', [({'degree': (-1, 2)}, 'degree=\\(min_degree, max_degree\\) must'), ({'degree': (0, 1.5)}, 'degree=\\(min_degree, max_degree\\) must'), ({'degree': (3, 2)}, 'degree=\\(min_degree, max_degree\\) must'), ({'degree': (1, 2, 3)}, 'int or tuple \\(min_degree, max_degree\\)')])
def test_polynomial_features_input_validation(params, err_msg):
    """Test that we raise errors for invalid input in PolynomialFeatures."""
    X = [[1], [2]]
    with pytest.raises(ValueError, match=err_msg):
        PolynomialFeatures(**params).fit(X)