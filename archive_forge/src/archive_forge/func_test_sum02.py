import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_sum02():
    for type in types:
        input = np.zeros([0, 4], type)
        output = ndimage.sum(input)
        assert_equal(output, 0.0)