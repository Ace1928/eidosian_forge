import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('mode', ['mirror', 'reflect', 'grid-mirror', 'grid-wrap', 'grid-constant', 'nearest'])
@pytest.mark.parametrize('order', range(6))
def test_boundary_spline_accuracy(self, mode, order):
    """Tests based on examples from gh-2640"""
    data = numpy.arange(-6, 7, dtype=float)
    x = numpy.linspace(-8, 15, num=1000)
    y = ndimage.map_coordinates(data, [x], order=order, mode=mode)
    npad = 32
    pad_mode = ndimage_to_numpy_mode.get(mode)
    padded = numpy.pad(data, npad, mode=pad_mode)
    expected = ndimage.map_coordinates(padded, [npad + x], order=order, mode=mode)
    atol = 1e-05 if mode == 'grid-constant' else 1e-12
    assert_allclose(y, expected, rtol=1e-07, atol=atol)