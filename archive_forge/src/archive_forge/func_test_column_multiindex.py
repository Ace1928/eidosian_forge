import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_column_multiindex(setup_path):
    index = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')], names=['first', 'second'])
    df = DataFrame(np.arange(12).reshape(3, 4), columns=index)
    expected = df.set_axis(df.index.to_numpy())
    with ensure_clean_store(setup_path) as store:
        store.put('df', df)
        tm.assert_frame_equal(store['df'], expected, check_index_type=True, check_column_type=True)
        store.put('df1', df, format='table')
        tm.assert_frame_equal(store['df1'], expected, check_index_type=True, check_column_type=True)
        msg = re.escape("cannot use a multi-index on axis [1] with data_columns ['A']")
        with pytest.raises(ValueError, match=msg):
            store.put('df2', df, format='table', data_columns=['A'])
        msg = re.escape('cannot use a multi-index on axis [1] with data_columns True')
        with pytest.raises(ValueError, match=msg):
            store.put('df3', df, format='table', data_columns=True)
    with ensure_clean_store(setup_path) as store:
        store.append('df2', df)
        store.append('df2', df)
        tm.assert_frame_equal(store['df2'], concat((df, df)))
    df = DataFrame(np.arange(12).reshape(3, 4), columns=Index(list('ABCD'), name='foo'))
    expected = df.set_axis(df.index.to_numpy())
    with ensure_clean_store(setup_path) as store:
        store.put('df1', df, format='table')
        tm.assert_frame_equal(store['df1'], expected, check_index_type=True, check_column_type=True)