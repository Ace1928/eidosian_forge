import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_histogram03():
    labels = [1, 0, 1, 1, 2, 2, 2, 2]
    expected1 = [0, 1, 0, 1, 1]
    expected2 = [0, 0, 0, 3, 0]
    input = np.array([1, 1, 3, 4, 3, 5, 3, 3])
    output = ndimage.histogram(input, 0, 4, 5, labels, (1, 2))
    assert_array_almost_equal(output[0], expected1)
    assert_array_almost_equal(output[1], expected2)