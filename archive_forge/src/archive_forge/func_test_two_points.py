import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_two_points(self):
    x = np.linspace(0, 1, 11)
    p = pchip([0, 1], [0, 2])
    assert_allclose(p(x), 2 * x, atol=1e-15)