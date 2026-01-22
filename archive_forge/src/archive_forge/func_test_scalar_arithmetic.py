import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
@pytest.mark.parametrize('size', [100, (2, 100), (100, 2), (1, 100), (100, 1)])
def test_scalar_arithmetic(size):
    numpy_arr = numpy.random.randint(-100, 100, size=size)
    modin_arr = np.array(numpy_arr)
    scalar = numpy.random.randint(1, 100)
    assert_scalar_or_array_equal(scalar * modin_arr, scalar * numpy_arr, err_msg='__mul__ failed.')
    assert_scalar_or_array_equal(modin_arr * scalar, scalar * numpy_arr, err_msg='__rmul__ failed.')
    assert_scalar_or_array_equal(scalar / modin_arr, scalar / numpy_arr, err_msg='__rtruediv__ failed.')
    assert_scalar_or_array_equal(modin_arr / scalar, numpy_arr / scalar, err_msg='__truediv__ failed.')
    assert_scalar_or_array_equal(scalar + modin_arr, scalar + numpy_arr, err_msg='__radd__ failed.')
    assert_scalar_or_array_equal(modin_arr + scalar, scalar + numpy_arr, err_msg='__add__ failed.')
    assert_scalar_or_array_equal(scalar - modin_arr, scalar - numpy_arr, err_msg='__rsub__ failed.')
    assert_scalar_or_array_equal(modin_arr - scalar, numpy_arr - scalar, err_msg='__sub__ failed.')