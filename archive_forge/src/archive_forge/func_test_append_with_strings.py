import datetime
from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
def test_append_with_strings(setup_path):
    with ensure_clean_store(setup_path) as store:

        def check_col(key, name, size):
            assert getattr(store.get_storer(key).table.description, name).itemsize == size
        df = DataFrame([[123, 'asdqwerty'], [345, 'dggnhebbsdfbdfb']])
        store.append('df_big', df)
        tm.assert_frame_equal(store.select('df_big'), df)
        check_col('df_big', 'values_block_1', 15)
        df2 = DataFrame([[124, 'asdqy'], [346, 'dggnhefbdfb']])
        store.append('df_big', df2)
        expected = concat([df, df2])
        tm.assert_frame_equal(store.select('df_big'), expected)
        check_col('df_big', 'values_block_1', 15)
        df = DataFrame([[123, 'asdqwerty'], [345, 'dggnhebbsdfbdfb']])
        store.append('df_big2', df, min_itemsize={'values': 50})
        tm.assert_frame_equal(store.select('df_big2'), df)
        check_col('df_big2', 'values_block_1', 50)
        store.append('df_new', df)
        df_new = DataFrame([[124, 'abcdefqhij'], [346, 'abcdefghijklmnopqrtsuvwxyz']])
        msg = 'Trying to store a string with len \\[26\\] in \\[values_block_1\\] column but\\nthis column has a limit of \\[15\\]!\\nConsider using min_itemsize to preset the sizes on these columns'
        with pytest.raises(ValueError, match=msg):
            store.append('df_new', df_new)
        df = DataFrame({'A': [0.0, 1.0, 2.0, 3.0, 4.0], 'B': [0.0, 1.0, 0.0, 1.0, 0.0], 'C': Index(['foo1', 'foo2', 'foo3', 'foo4', 'foo5'], dtype=object), 'D': date_range('20130101', periods=5)}).set_index('C')
        store.append('ss', df['B'], min_itemsize={'index': 4})
        tm.assert_series_equal(store.select('ss'), df['B'])
        store.append('ss2', df['B'], data_columns=True, min_itemsize={'index': 4})
        tm.assert_series_equal(store.select('ss2'), df['B'])
        store.put('ss3', df, format='table', min_itemsize={'index': 6})
        df2 = df.copy().reset_index().assign(C='longer').set_index('C')
        store.append('ss3', df2)
        tm.assert_frame_equal(store.select('ss3'), concat([df, df2]))
        store.put('ss4', df['B'], format='table', min_itemsize={'index': 6})
        store.append('ss4', df2['B'])
        tm.assert_series_equal(store.select('ss4'), concat([df['B'], df2['B']]))
        _maybe_remove(store, 'df')
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        df['string'] = 'foo'
        df.loc[df.index[1:4], 'string'] = np.nan
        df['string2'] = 'bar'
        df.loc[df.index[4:8], 'string2'] = np.nan
        df['string3'] = 'bah'
        df.loc[df.index[1:], 'string3'] = np.nan
        store.append('df', df)
        result = store.select('df')
        tm.assert_frame_equal(result, df)
    with ensure_clean_store(setup_path) as store:
        df = DataFrame({'A': 'foo', 'B': 'bar'}, index=range(10))
        _maybe_remove(store, 'df')
        store.append('df', df, min_itemsize={'A': 200})
        check_col('df', 'A', 200)
        assert store.get_storer('df').data_columns == ['A']
        _maybe_remove(store, 'df')
        store.append('df', df, data_columns=['B'], min_itemsize={'A': 200})
        check_col('df', 'A', 200)
        assert store.get_storer('df').data_columns == ['B', 'A']
        _maybe_remove(store, 'df')
        store.append('df', df, data_columns=['B'], min_itemsize={'values': 200})
        check_col('df', 'B', 200)
        check_col('df', 'values_block_0', 200)
        assert store.get_storer('df').data_columns == ['B']
        _maybe_remove(store, 'df')
        store.append('df', df[:5], min_itemsize=200)
        store.append('df', df[5:], min_itemsize=200)
        tm.assert_frame_equal(store['df'], df)
        df = DataFrame(['foo', 'foo', 'foo', 'barh', 'barh', 'barh'], columns=['A'])
        _maybe_remove(store, 'df')
        msg = re.escape('min_itemsize has the key [foo] which is not an axis or data_column')
        with pytest.raises(ValueError, match=msg):
            store.append('df', df, min_itemsize={'foo': 20, 'foobar': 20})