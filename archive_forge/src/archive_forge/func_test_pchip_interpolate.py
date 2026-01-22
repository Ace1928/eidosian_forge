import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_pchip_interpolate(self):
    assert_array_almost_equal(pchip_interpolate([1, 2, 3], [4, 5, 6], [0.5], der=1), [1.0])
    assert_array_almost_equal(pchip_interpolate([1, 2, 3], [4, 5, 6], [0.5], der=0), [3.5])
    assert_array_almost_equal(pchip_interpolate([1, 2, 3], [4, 5, 6], [0.5], der=[0, 1]), [[3.5], [1]])