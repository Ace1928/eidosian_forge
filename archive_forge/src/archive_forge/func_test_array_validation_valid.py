import numpy
import pytest
from thinc.types import (
@pytest.mark.parametrize('arr,arr_type', [(numpy.zeros(0, dtype=numpy.float32), Floats1d), (numpy.zeros((0, 0), dtype=numpy.float32), Floats2d), (numpy.zeros((0, 0, 0), dtype=numpy.float32), Floats3d), (numpy.zeros((0, 0, 0, 0), dtype=numpy.float32), Floats4d), (numpy.zeros(0, dtype=numpy.int32), Ints1d), (numpy.zeros((0, 0), dtype=numpy.int32), Ints2d), (numpy.zeros((0, 0, 0), dtype=numpy.int32), Ints3d), (numpy.zeros((0, 0, 0, 0), dtype=numpy.int32), Ints4d)])
def test_array_validation_valid(arr, arr_type):
    test_model = create_model('TestModel', arr=(arr_type, ...))
    result = test_model(arr=arr)
    assert numpy.array_equal(arr, result.arr)