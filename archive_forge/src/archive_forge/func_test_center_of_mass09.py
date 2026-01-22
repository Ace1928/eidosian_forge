import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_center_of_mass09():
    labels = [1, 2]
    expected = [(0.5, 0.0), (0.5, 1.0)]
    input = np.array([[1, 2], [1, 1]], bool)
    output = ndimage.center_of_mass(input, labels, [1, 2])
    assert_array_almost_equal(output, expected)