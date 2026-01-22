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
def test_gauss_memory_overlap(self):
    input = numpy.arange(100 * 100).astype(numpy.float32)
    input.shape = (100, 100)
    output1 = ndimage.gaussian_filter(input, 1.0)
    ndimage.gaussian_filter(input, 1.0, output=input)
    assert_array_almost_equal(output1, input)