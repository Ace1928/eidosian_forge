import numpy as np
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_get_numeric_data(self, using_infer_string):
    datetime64name = np.dtype('M8[s]').name
    objectname = np.dtype(np.object_).name
    df = DataFrame({'a': 1.0, 'b': 2, 'c': 'foo', 'f': Timestamp('20010102')}, index=np.arange(10))
    result = df.dtypes
    expected = Series([np.dtype('float64'), np.dtype('int64'), np.dtype(objectname) if not using_infer_string else 'string', np.dtype(datetime64name)], index=['a', 'b', 'c', 'f'])
    tm.assert_series_equal(result, expected)
    df = DataFrame({'a': 1.0, 'b': 2, 'c': 'foo', 'd': np.array([1.0] * 10, dtype='float32'), 'e': np.array([1] * 10, dtype='int32'), 'f': np.array([1] * 10, dtype='int16'), 'g': Timestamp('20010102')}, index=np.arange(10))
    result = df._get_numeric_data()
    expected = df.loc[:, ['a', 'b', 'd', 'e', 'f']]
    tm.assert_frame_equal(result, expected)
    only_obj = df.loc[:, ['c', 'g']]
    result = only_obj._get_numeric_data()
    expected = df.loc[:, []]
    tm.assert_frame_equal(result, expected)
    df = DataFrame.from_dict({'a': [1, 2], 'b': ['foo', 'bar'], 'c': [np.pi, np.e]})
    result = df._get_numeric_data()
    expected = DataFrame.from_dict({'a': [1, 2], 'c': [np.pi, np.e]})
    tm.assert_frame_equal(result, expected)
    df = result.copy()
    result = df._get_numeric_data()
    expected = df
    tm.assert_frame_equal(result, expected)