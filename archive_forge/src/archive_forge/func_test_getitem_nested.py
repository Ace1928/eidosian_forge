import numpy
import pytest
from pandas.core.dtypes.common import is_list_like
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
def test_getitem_nested():
    data = [1, 2, 3, 4, 5]
    numpy_result = numpy.array(data)[1:3][1]
    modin_result = np.array(data)[1:3][1]
    if is_list_like(numpy_result):
        assert_scalar_or_array_equal(modin_result, numpy_result)
        assert modin_result.shape == numpy_result.shape
    else:
        assert modin_result == numpy_result
    data = [[1, 2, 3], [4, 5, 6]]
    numpy_result = numpy.array(data)[1][1]
    modin_result = np.array(data)[1][1]
    if is_list_like(numpy_result):
        assert_scalar_or_array_equal(modin_result, numpy_result)
        assert modin_result.shape == numpy_result.shape
    else:
        assert modin_result == numpy_result