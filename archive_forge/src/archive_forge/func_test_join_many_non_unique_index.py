import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_many_non_unique_index(self):
    df1 = DataFrame({'a': [1, 1], 'b': [1, 1], 'c': [10, 20]})
    df2 = DataFrame({'a': [1, 1], 'b': [1, 2], 'd': [100, 200]})
    df3 = DataFrame({'a': [1, 1], 'b': [1, 2], 'e': [1000, 2000]})
    idf1 = df1.set_index(['a', 'b'])
    idf2 = df2.set_index(['a', 'b'])
    idf3 = df3.set_index(['a', 'b'])
    result = idf1.join([idf2, idf3], how='outer')
    df_partially_merged = merge(df1, df2, on=['a', 'b'], how='outer')
    expected = merge(df_partially_merged, df3, on=['a', 'b'], how='outer')
    result = result.reset_index()
    expected = expected[result.columns]
    expected['a'] = expected.a.astype('int64')
    expected['b'] = expected.b.astype('int64')
    tm.assert_frame_equal(result, expected)
    df1 = DataFrame({'a': [1, 1, 1], 'b': [1, 1, 1], 'c': [10, 20, 30]})
    df2 = DataFrame({'a': [1, 1, 1], 'b': [1, 1, 2], 'd': [100, 200, 300]})
    df3 = DataFrame({'a': [1, 1, 1], 'b': [1, 1, 2], 'e': [1000, 2000, 3000]})
    idf1 = df1.set_index(['a', 'b'])
    idf2 = df2.set_index(['a', 'b'])
    idf3 = df3.set_index(['a', 'b'])
    result = idf1.join([idf2, idf3], how='inner')
    df_partially_merged = merge(df1, df2, on=['a', 'b'], how='inner')
    expected = merge(df_partially_merged, df3, on=['a', 'b'], how='inner')
    result = result.reset_index()
    tm.assert_frame_equal(result, expected.loc[:, result.columns])
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'], 'C': np.random.default_rng(2).standard_normal(8), 'D': np.random.default_rng(2).standard_normal(8)})
    s = Series(np.repeat(np.arange(8), 2), index=np.repeat(np.arange(8), 2), name='TEST')
    inner = df.join(s, how='inner')
    outer = df.join(s, how='outer')
    left = df.join(s, how='left')
    right = df.join(s, how='right')
    tm.assert_frame_equal(inner, outer)
    tm.assert_frame_equal(inner, left)
    tm.assert_frame_equal(inner, right)