import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_affine_transform27(self):
    data = numpy.array([[4, 1, 3, 2], [7, 6, 8, 5], [3, 5, 3, 6]])
    tform_h1 = numpy.hstack((numpy.eye(2), -numpy.ones((2, 1))))
    tform_h2 = numpy.vstack((tform_h1, [[5, 2, 1]]))
    assert_raises(ValueError, ndimage.affine_transform, data, tform_h2)