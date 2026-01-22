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
@pytest.mark.parametrize(['bias', 'intercept'], [(True, False), (False, True)])
@pytest.mark.parametrize('degree', [1, 2, 3, 4, 5])
def test_spline_transformer_extrapolation(bias, intercept, degree):
    """Test that B-spline extrapolation works correctly."""
    X = np.linspace(-1, 1, 100)[:, None]
    y = X.squeeze()
    pipe = Pipeline([['spline', SplineTransformer(n_knots=4, degree=degree, include_bias=bias, extrapolation='constant')], ['ols', LinearRegression(fit_intercept=intercept)]])
    pipe.fit(X, y)
    assert_allclose(pipe.predict([[-10], [5]]), [-1, 1])
    pipe = Pipeline([['spline', SplineTransformer(n_knots=4, degree=degree, include_bias=bias, extrapolation='linear')], ['ols', LinearRegression(fit_intercept=intercept)]])
    pipe.fit(X, y)
    assert_allclose(pipe.predict([[-10], [5]]), [-10, 5])
    splt = SplineTransformer(n_knots=4, degree=degree, include_bias=bias, extrapolation='error')
    splt.fit(X)
    msg = 'X contains values beyond the limits of the knots'
    with pytest.raises(ValueError, match=msg):
        splt.transform([[-10]])
    with pytest.raises(ValueError, match=msg):
        splt.transform([[5]])