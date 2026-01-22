import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_extrema01():
    labels = np.array([1, 0], bool)
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output1 = ndimage.extrema(input, labels=labels)
        output2 = ndimage.minimum(input, labels=labels)
        output3 = ndimage.maximum(input, labels=labels)
        output4 = ndimage.minimum_position(input, labels=labels)
        output5 = ndimage.maximum_position(input, labels=labels)
        assert_equal(output1, (output2, output3, output4, output5))