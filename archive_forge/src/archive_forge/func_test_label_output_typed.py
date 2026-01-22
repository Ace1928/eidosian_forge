import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_label_output_typed():
    data = np.ones([5])
    for t in types:
        output = np.zeros([5], dtype=t)
        n = ndimage.label(data, output=output)
        assert_array_almost_equal(output, 1)
        assert_equal(n, 1)