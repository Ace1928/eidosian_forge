import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_affine_transform_multi_d_endianness_with_output_parameter(self):
    data = numpy.array([1])
    for out in [data.dtype, data.dtype.newbyteorder(), numpy.empty_like(data), numpy.empty_like(data).astype(data.dtype.newbyteorder())]:
        returned = ndimage.affine_transform(data, [[1]], output=out)
        result = out if returned is None else returned
        assert_array_almost_equal(result, [1])