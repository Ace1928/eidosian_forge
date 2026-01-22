import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_affine_transform26(self, order):
    data = numpy.array([[4, 1, 3, 2], [7, 6, 8, 5], [3, 5, 3, 6]])
    if order > 1:
        filtered = ndimage.spline_filter(data, order=order)
    else:
        filtered = data
    tform_original = numpy.eye(2)
    offset_original = -numpy.ones((2, 1))
    tform_h1 = numpy.hstack((tform_original, offset_original))
    tform_h2 = numpy.vstack((tform_h1, [[0, 0, 1]]))
    out1 = ndimage.affine_transform(filtered, tform_original, offset_original.ravel(), order=order, prefilter=False)
    out2 = ndimage.affine_transform(filtered, tform_h1, order=order, prefilter=False)
    out3 = ndimage.affine_transform(filtered, tform_h2, order=order, prefilter=False)
    for out in [out1, out2, out3]:
        assert_array_almost_equal(out, [[0, 0, 0, 0], [0, 4, 1, 3], [0, 7, 6, 8]])