import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.interpolate import (griddata, NearestNDInterpolator,
def test_1d_borders(self):
    x = np.array([1, 2.5, 3, 4.5, 5, 6])
    y = np.array([1, 2, 0, 3.9, 2, 1])
    xi = np.array([0.9, 6.5])
    yi_should = np.array([1.0, 1.0])
    method = 'nearest'
    assert_allclose(griddata(x, y, xi, method=method), yi_should, err_msg=method, atol=1e-14)
    assert_allclose(griddata(x.reshape(6, 1), y, xi, method=method), yi_should, err_msg=method, atol=1e-14)
    assert_allclose(griddata((x,), y, (xi,), method=method), yi_should, err_msg=method, atol=1e-14)