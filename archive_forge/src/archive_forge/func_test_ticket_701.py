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
def test_ticket_701():
    arr = numpy.arange(4).reshape((2, 2))

    def func(x):
        return numpy.min(x)
    res = ndimage.generic_filter(arr, func, size=(1, 1))
    res2 = ndimage.generic_filter(arr, func, size=1)
    assert_equal(res, res2)