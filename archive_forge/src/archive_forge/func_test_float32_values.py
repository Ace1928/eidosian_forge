import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize('xi_dtype', [np.float32, np.float64])
def test_float32_values(self, dtype, xi_dtype):

    def f(x, y):
        return 2 * x ** 3 + 3 * y ** 2
    x = np.linspace(1, 4, 11)
    y = np.linspace(4, 7, 22)
    xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)
    data = f(xg, yg)
    data = data.astype(dtype)
    interp = RegularGridInterpolator((x, y), data)
    pts = np.array([[2.1, 6.2], [3.3, 5.2]], dtype=xi_dtype)
    assert_allclose(interp(pts), [134.10469388, 153.40069388], atol=1e-07)