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
def test_correlate_mode_sequence(self):
    kernel = numpy.ones((2, 2))
    array = numpy.ones((3, 3), float)
    with assert_raises(RuntimeError):
        ndimage.correlate(array, kernel, mode=['nearest', 'reflect'])
    with assert_raises(RuntimeError):
        ndimage.convolve(array, kernel, mode=['nearest', 'reflect'])