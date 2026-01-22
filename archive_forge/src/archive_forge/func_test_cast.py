import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_cast(self):
    data = np.array([[0, 4, 12, 27, 47, 60, 79, 87, 99, 100], [-33, -33, -19, -2, 12, 26, 38, 45, 53, 55]])
    xx = np.arange(100)
    curve = pchip(data[0], data[1])(xx)
    data1 = data * 1.0
    curve1 = pchip(data1[0], data1[1])(xx)
    assert_allclose(curve, curve1, atol=1e-14, rtol=1e-14)