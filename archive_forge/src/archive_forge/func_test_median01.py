import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_median01():
    a = np.array([[1, 2, 0, 1], [5, 3, 0, 4], [0, 0, 0, 7], [9, 3, 0, 0]])
    labels = np.array([[1, 1, 0, 2], [1, 1, 0, 2], [0, 0, 0, 2], [3, 3, 0, 0]])
    output = ndimage.median(a, labels=labels, index=[1, 2, 3])
    assert_array_almost_equal(output, [2.5, 4.0, 6.0])