import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
def test_string_select(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        df['x'] = 'none'
        df.loc[df.index[2:7], 'x'] = ''
        store.append('df', df, data_columns=['x'])
        result = store.select('df', 'x=none')
        expected = df[df.x == 'none']
        tm.assert_frame_equal(result, expected)
        result = store.select('df', 'x!=none')
        expected = df[df.x != 'none']
        tm.assert_frame_equal(result, expected)
        df2 = df.copy()
        df2.loc[df2.x == '', 'x'] = np.nan
        store.append('df2', df2, data_columns=['x'])
        result = store.select('df2', 'x!=none')
        expected = df2[isna(df2.x)]
        tm.assert_frame_equal(result, expected)
        df['int'] = 1
        df.loc[df.index[2:7], 'int'] = 2
        store.append('df3', df, data_columns=['int'])
        result = store.select('df3', 'int=2')
        expected = df[df.int == 2]
        tm.assert_frame_equal(result, expected)
        result = store.select('df3', 'int!=2')
        expected = df[df.int != 2]
        tm.assert_frame_equal(result, expected)