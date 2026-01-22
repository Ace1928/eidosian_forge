import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_standard_deviation02():
    for type in types:
        input = np.array([1], type)
        output = ndimage.standard_deviation(input)
        assert_almost_equal(output, 0.0)