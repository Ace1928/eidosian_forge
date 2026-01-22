import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_center_of_mass04():
    expected = [1, 1]
    for type in types:
        input = np.array([[0, 0], [0, 1]], type)
        output = ndimage.center_of_mass(input)
        assert_array_almost_equal(output, expected)