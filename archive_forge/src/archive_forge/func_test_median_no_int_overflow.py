import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_median_no_int_overflow():
    a = np.asarray([65, 70], dtype=np.int8)
    output = ndimage.median(a, labels=np.ones((2,)), index=[1])
    assert_array_almost_equal(output, [67.5])