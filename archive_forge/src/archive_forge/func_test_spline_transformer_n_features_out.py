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
@pytest.mark.parametrize('n_knots', [5, 10])
@pytest.mark.parametrize('include_bias', [True, False])
@pytest.mark.parametrize('degree', [3, 4])
@pytest.mark.parametrize('extrapolation', ['error', 'constant', 'linear', 'continue', 'periodic'])
@pytest.mark.parametrize('sparse_output', [False, True])
def test_spline_transformer_n_features_out(n_knots, include_bias, degree, extrapolation, sparse_output):
    """Test that transform results in n_features_out_ features."""
    if sparse_output and sp_version < parse_version('1.8.0'):
        pytest.skip('The option `sparse_output` is available as of scipy 1.8.0')
    splt = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=include_bias, extrapolation=extrapolation, sparse_output=sparse_output)
    X = np.linspace(0, 1, 10)[:, None]
    splt.fit(X)
    assert splt.transform(X).shape[1] == splt.n_features_out_