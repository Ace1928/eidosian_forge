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
def test_minmax_filter(self):
    d = numpy.random.randn(500, 500)
    os = numpy.empty([4] + list(d.shape))
    ot = numpy.empty_like(os)
    self.check_func_serial(4, ndimage.maximum_filter, (d, 3), os)
    self.check_func_thread(4, ndimage.maximum_filter, (d, 3), ot)
    assert_array_equal(os, ot)
    self.check_func_serial(4, ndimage.minimum_filter, (d, 3), os)
    self.check_func_thread(4, ndimage.minimum_filter, (d, 3), ot)
    assert_array_equal(os, ot)