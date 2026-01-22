import warnings
import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
def test_array_where():
    numpy_flat_arr = numpy.random.randint(-100, 100, size=100)
    modin_flat_arr = np.array(numpy_flat_arr)
    with pytest.warns(UserWarning, match='np.where method with only condition specified'):
        warnings.filterwarnings('ignore', message='Distributing')
        (modin_flat_arr <= 0).where()
    with pytest.raises(ValueError, match='np.where requires x and y'):
        (modin_flat_arr <= 0).where(x=['Should Fail.'])
    with pytest.warns(UserWarning, match='np.where not supported when both x and y'):
        warnings.filterwarnings('ignore', message='Distributing')
        modin_result = (modin_flat_arr <= 0).where(x=4, y=5)
    numpy_result = numpy.where(numpy_flat_arr <= 0, 4, 5)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_flat_bool_arr = modin_flat_arr <= 0
    numpy_flat_bool_arr = numpy_flat_arr <= 0
    modin_result = modin_flat_bool_arr.where(x=5, y=modin_flat_arr)
    numpy_result = numpy.where(numpy_flat_bool_arr, 5, numpy_flat_arr)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_flat_bool_arr.where(x=modin_flat_arr, y=5)
    numpy_result = numpy.where(numpy_flat_bool_arr, numpy_flat_arr, 5)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_flat_bool_arr.where(x=modin_flat_arr, y=-1 * modin_flat_arr)
    numpy_result = numpy.where(numpy_flat_bool_arr, numpy_flat_arr, -1 * numpy_flat_arr)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    modin_bool_arr = modin_arr > 0
    numpy_bool_arr = numpy_arr > 0
    modin_result = modin_bool_arr.where(modin_arr, 10 * modin_arr)
    numpy_result = numpy.where(numpy_bool_arr, numpy_arr, 10 * numpy_arr)
    assert_scalar_or_array_equal(modin_result, numpy_result)