import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
def test_select_iterator(tmp_path, setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        _maybe_remove(store, 'df')
        store.append('df', df)
        expected = store.select('df')
        results = list(store.select('df', iterator=True))
        result = concat(results)
        tm.assert_frame_equal(expected, result)
        results = list(store.select('df', chunksize=2))
        assert len(results) == 5
        result = concat(results)
        tm.assert_frame_equal(expected, result)
        results = list(store.select('df', chunksize=2))
        result = concat(results)
        tm.assert_frame_equal(result, expected)
    path = tmp_path / setup_path
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    df.to_hdf(path, key='df_non_table')
    msg = 'can only use an iterator or chunksize on a table'
    with pytest.raises(TypeError, match=msg):
        read_hdf(path, 'df_non_table', chunksize=2)
    with pytest.raises(TypeError, match=msg):
        read_hdf(path, 'df_non_table', iterator=True)
    path = tmp_path / setup_path
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    df.to_hdf(path, key='df', format='table')
    results = list(read_hdf(path, 'df', chunksize=2))
    result = concat(results)
    assert len(results) == 5
    tm.assert_frame_equal(result, df)
    tm.assert_frame_equal(result, read_hdf(path, 'df'))
    with ensure_clean_store(setup_path) as store:
        df1 = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        store.append('df1', df1, data_columns=True)
        df2 = df1.copy().rename(columns='{}_2'.format)
        df2['foo'] = 'bar'
        store.append('df2', df2)
        df = concat([df1, df2], axis=1)
        expected = store.select_as_multiple(['df1', 'df2'], selector='df1')
        results = list(store.select_as_multiple(['df1', 'df2'], selector='df1', chunksize=2))
        result = concat(results)
        tm.assert_frame_equal(expected, result)