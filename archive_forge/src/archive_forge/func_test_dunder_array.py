import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('array', [True, False])
def test_dunder_array(array):
    obj = PeriodIndex(['2000-01-01', '2001-01-01'], freq='D')
    if array:
        obj = obj._data
    expected = np.array([obj[0], obj[1]], dtype=object)
    result = np.array(obj)
    tm.assert_numpy_array_equal(result, expected)
    result = np.asarray(obj)
    tm.assert_numpy_array_equal(result, expected)
    expected = obj.asi8
    for dtype in ['i8', 'int64', np.int64]:
        result = np.array(obj, dtype=dtype)
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(obj, dtype=dtype)
        tm.assert_numpy_array_equal(result, expected)
    for dtype in ['float64', 'int32', 'uint64']:
        msg = 'argument must be'
        with pytest.raises(TypeError, match=msg):
            np.array(obj, dtype=dtype)
        with pytest.raises(TypeError, match=msg):
            np.array(obj, dtype=getattr(np, dtype))