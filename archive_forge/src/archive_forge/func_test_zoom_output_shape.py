import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_zoom_output_shape():
    """Ticket #643"""
    x = numpy.arange(12).reshape((3, 4))
    ndimage.zoom(x, 2, output=numpy.zeros((6, 8)))