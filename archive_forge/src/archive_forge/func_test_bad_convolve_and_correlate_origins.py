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
def test_bad_convolve_and_correlate_origins():
    """Regression test for gh-822."""
    assert_raises(ValueError, ndimage.correlate1d, [0, 1, 2, 3, 4, 5], [1, 1, 2, 0], origin=2)
    assert_raises(ValueError, ndimage.correlate, [0, 1, 2, 3, 4, 5], [0, 1, 2], origin=[2])
    assert_raises(ValueError, ndimage.correlate, numpy.ones((3, 5)), numpy.ones((2, 2)), origin=[0, 1])
    assert_raises(ValueError, ndimage.convolve1d, numpy.arange(10), numpy.ones(3), origin=-2)
    assert_raises(ValueError, ndimage.convolve, numpy.arange(10), numpy.ones(3), origin=[-2])
    assert_raises(ValueError, ndimage.convolve, numpy.ones((3, 5)), numpy.ones((2, 2)), origin=[0, -2])