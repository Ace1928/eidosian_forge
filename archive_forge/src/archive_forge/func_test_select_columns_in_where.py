import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
def test_select_columns_in_where(setup_path):
    index = MultiIndex(levels=[['foo', 'bar', 'baz', 'qux'], ['one', 'two', 'three']], codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]], names=['foo_name', 'bar_name'])
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), index=index, columns=['A', 'B', 'C'])
    with ensure_clean_store(setup_path) as store:
        store.put('df', df, format='table')
        expected = df[['A']]
        tm.assert_frame_equal(store.select('df', columns=['A']), expected)
        tm.assert_frame_equal(store.select('df', where="columns=['A']"), expected)
    s = Series(np.random.default_rng(2).standard_normal(10), index=index, name='A')
    with ensure_clean_store(setup_path) as store:
        store.put('s', s, format='table')
        tm.assert_series_equal(store.select('s', where="columns=['A']"), s)