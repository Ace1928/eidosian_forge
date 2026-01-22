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
def test_multiple_modes_gaussian_laplace():
    arr = numpy.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    expected = numpy.array([[-0.28438687, 0.01559809, 0.19773499], [-0.36630503, -0.20069774, 0.0748362], [0.15849176, 0.18495566, 0.21934094]])
    modes = ['reflect', 'wrap']
    assert_almost_equal(expected, ndimage.gaussian_laplace(arr, 1, mode=modes))