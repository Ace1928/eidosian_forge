import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.gam.smooth_basis import (UnivariatePolynomialSmoother,
@pytest.mark.parametrize('x, df, degree', [(np.c_[np.linspace(0, 1, 100), np.linspace(0, 10, 100)], [5, 6], [3, 5]), (np.linspace(0, 1, 100), 6, 3)])
def test_bsplines(x, df, degree):
    bspline = BSplines(x, df, degree)
    bspline.transform(x)