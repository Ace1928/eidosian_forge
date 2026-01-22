import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_affine_transform_with_string_output(self):
    data = numpy.array([1])
    out = ndimage.affine_transform(data, [[1]], output='f')
    assert_(out.dtype is numpy.dtype('f'))
    assert_array_almost_equal(out, [1])