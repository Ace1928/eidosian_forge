import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
@pytest.mark.parametrize('data', [small_arr_r_2d, small_arr_r_1d], ids=['2D', '1D'])
def test_logical_not(data):
    x1 = data
    numpy_result = numpy.logical_not(x1)
    x1 = np.array(x1)
    modin_result = np.logical_not(x1)
    assert_scalar_or_array_equal(modin_result, numpy_result)