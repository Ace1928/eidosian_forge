import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_variance04():
    input = np.array([1, 0], bool)
    output = ndimage.variance(input)
    assert_almost_equal(output, 0.25)