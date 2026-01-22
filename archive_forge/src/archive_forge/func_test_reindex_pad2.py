import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_pad2():
    s = Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
    new_index = ['a', 'g', 'c', 'f']
    expected = Series([1, 1, 3, 3], index=new_index)
    result = s.reindex(new_index).ffill()
    tm.assert_series_equal(result, expected.astype('float64'))
    msg = "The 'downcast' keyword in ffill is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.reindex(new_index).ffill(downcast='infer')
    tm.assert_series_equal(result, expected)
    expected = Series([1, 5, 3, 5], index=new_index)
    result = s.reindex(new_index, method='ffill')
    tm.assert_series_equal(result, expected)