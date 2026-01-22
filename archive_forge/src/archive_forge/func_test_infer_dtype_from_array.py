from datetime import (
import numpy as np
import pytest
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import is_dtype_equal
from pandas import (
@pytest.mark.parametrize('arr, expected', [([1], np.dtype(int)), (np.array([1], dtype=np.int64), np.int64), ([np.nan, 1, ''], np.object_), (np.array([[1.0, 2.0]]), np.float64), (Categorical(list('aabc')), 'category'), (Categorical([1, 2, 3]), 'category'), (date_range('20160101', periods=3), np.dtype('=M8[ns]')), (date_range('20160101', periods=3, tz='US/Eastern'), 'datetime64[ns, US/Eastern]'), (Series([1.0, 2, 3]), np.float64), (Series(list('abc')), np.object_), (Series(date_range('20160101', periods=3, tz='US/Eastern')), 'datetime64[ns, US/Eastern]')])
def test_infer_dtype_from_array(arr, expected, using_infer_string):
    dtype, _ = infer_dtype_from_array(arr)
    if using_infer_string and isinstance(arr, Series) and (arr.tolist() == ['a', 'b', 'c']):
        expected = 'string'
    assert is_dtype_equal(dtype, expected)