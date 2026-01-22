import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_zoom_output_shape_roundoff(self):
    arr = numpy.zeros((3, 11, 25))
    zoom = (4.0 / 3, 15.0 / 11, 29.0 / 25)
    out = ndimage.zoom(arr, zoom)
    assert_array_equal(out.shape, (4, 15, 29))