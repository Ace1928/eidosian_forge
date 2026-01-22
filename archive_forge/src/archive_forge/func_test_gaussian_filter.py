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
def test_gaussian_filter():
    data = numpy.array([1], dtype=numpy.float16)
    sigma = 1.0
    with assert_raises(RuntimeError):
        ndimage.gaussian_filter(data, sigma)