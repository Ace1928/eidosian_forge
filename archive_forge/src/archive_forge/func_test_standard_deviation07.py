import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_standard_deviation07():
    labels = [1]
    with np.errstate(all='ignore'):
        for type in types:
            input = np.array([-0.00619519], type)
            output = ndimage.standard_deviation(input, labels, [1])
            assert_array_almost_equal(output, [0])