import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_affine_transform_output_shape(self):
    data = numpy.arange(8, dtype=numpy.float64)
    out = numpy.ones((16,))
    ndimage.affine_transform(data, [[1]], output=out)
    assert_array_almost_equal(out[:8], data)
    with pytest.raises(RuntimeError):
        ndimage.affine_transform(data, [[1]], output=out, output_shape=(12,))