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
def test_gauss01(self):
    input = numpy.array([[1, 2, 3], [2, 4, 6]], numpy.float32)
    output = ndimage.gaussian_filter(input, 0)
    assert_array_almost_equal(output, input)