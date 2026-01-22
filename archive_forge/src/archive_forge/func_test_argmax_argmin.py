import warnings
import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
@pytest.mark.parametrize('method', ['argmax', 'argmin'])
def test_argmax_argmin(method):
    numpy_arr = numpy.array([[1, 2, 3], [4, 5, np.NaN]])
    modin_arr = np.array(numpy_arr)
    assert_scalar_or_array_equal(getattr(np, method)(modin_arr, axis=1), getattr(numpy, method)(numpy_arr, axis=1))