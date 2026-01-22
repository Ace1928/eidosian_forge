import functools
import itertools
import math
import numpy
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d
from . import types, float_types, complex_types
def test_multiple_modes_gaussian_gradient_magnitude():
    arr = numpy.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    expected = numpy.array([[0.04928965, 0.09745625, 0.06405368], [0.23056905, 0.14025305, 0.04550846], [0.19894369, 0.1495006, 0.0679685]])
    modes = ['reflect', 'wrap']
    calculated = ndimage.gaussian_gradient_magnitude(arr, 1, mode=modes)
    assert_almost_equal(expected, calculated)