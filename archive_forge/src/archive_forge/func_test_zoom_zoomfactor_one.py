import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_zoom_zoomfactor_one(self):
    arr = numpy.zeros((1, 5, 5))
    zoom = (1.0, 2.0, 2.0)
    out = ndimage.zoom(arr, zoom, cval=7)
    ref = numpy.zeros((1, 10, 10))
    assert_array_almost_equal(out, ref)