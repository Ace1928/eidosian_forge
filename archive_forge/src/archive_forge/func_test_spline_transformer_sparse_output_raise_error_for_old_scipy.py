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
@pytest.mark.skipif(sp_version >= parse_version('1.8.0'), reason='The option `sparse_output` is available as of scipy 1.8.0')
def test_spline_transformer_sparse_output_raise_error_for_old_scipy():
    """Test that SplineTransformer with sparse=True raises for scipy<1.8.0."""
    X = [[1], [2]]
    with pytest.raises(ValueError, match='scipy>=1.8.0'):
        SplineTransformer(sparse_output=True).fit(X)