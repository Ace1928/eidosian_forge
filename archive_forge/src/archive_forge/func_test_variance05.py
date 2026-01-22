import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_variance05():
    labels = [2, 2, 3]
    for type in types:
        input = np.array([1, 3, 8], type)
        output = ndimage.variance(input, labels, 2)
        assert_almost_equal(output, 1.0)