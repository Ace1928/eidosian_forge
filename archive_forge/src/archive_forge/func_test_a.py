import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_a(self):
    x = [0, 1, 2, 6]
    labels = [0, 0, 1, 1]
    index = [0, 1]
    for shp in [(4,), (2, 2)]:
        x = np.array(x).reshape(shp)
        labels = np.array(labels).reshape(shp)
        counts, sums = ndimage._measurements._stats(x, labels=labels, index=index)
        assert_array_equal(counts, [2, 2])
        assert_array_equal(sums, [1.0, 8.0])