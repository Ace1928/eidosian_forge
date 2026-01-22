import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_affine_transform_1d_endianness_with_output_parameter(self):
    data = numpy.ones((2, 2))
    for out in [numpy.empty_like(data), numpy.empty_like(data).astype(data.dtype.newbyteorder()), data.dtype, data.dtype.newbyteorder()]:
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'The behavior of affine_transform with a 1-D array .* has changed')
            returned = ndimage.affine_transform(data, [1, 1], output=out)
        result = out if returned is None else returned
        assert_array_almost_equal(result, [[1, 1], [1, 1]])