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
@pytest.mark.parametrize('dtype', complex_types)
@pytest.mark.parametrize('dtype_output', complex_types)
def test_correlate1d_complex_input_and_kernel(self, dtype, dtype_output):
    kernel = numpy.array([1, 1 + 1j], dtype)
    array = numpy.array([1, 2j, 3, 1 + 4j, 5, 6j], dtype)
    self._validate_complex(array, kernel, dtype_output)