import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
def test_rem_mod():
    """Tests remainder and mod, which, unlike the C/matlab equivalents, are identical in numpy."""
    a = numpy.array([[2, -1], [10, -3]])
    b = numpy.array(([-3, 3], [3, -7]))
    numpy_result = numpy.remainder(a, b)
    modin_result = np.remainder(np.array(a), np.array(b))
    assert_scalar_or_array_equal(modin_result, numpy_result)
    numpy_result = numpy.mod(a, b)
    modin_result = np.mod(np.array(a), np.array(b))
    assert_scalar_or_array_equal(modin_result, numpy_result)