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
def test_append_all_nans(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame({'A1': np.random.default_rng(2).standard_normal(20), 'A2': np.random.default_rng(2).standard_normal(20)}, index=np.arange(20))
        df.loc[0:15, :] = np.nan
        _maybe_remove(store, 'df')
        store.append('df', df[:10], dropna=True)
        store.append('df', df[10:], dropna=True)
        tm.assert_frame_equal(store['df'], df[-4:], check_index_type=True)
        _maybe_remove(store, 'df2')
        store.append('df2', df[:10], dropna=False)
        store.append('df2', df[10:], dropna=False)
        tm.assert_frame_equal(store['df2'], df, check_index_type=True)
        with pd.option_context('io.hdf.dropna_table', False):
            _maybe_remove(store, 'df3')
            store.append('df3', df[:10])
            store.append('df3', df[10:])
            tm.assert_frame_equal(store['df3'], df)
        with pd.option_context('io.hdf.dropna_table', True):
            _maybe_remove(store, 'df4')
            store.append('df4', df[:10])
            store.append('df4', df[10:])
            tm.assert_frame_equal(store['df4'], df[-4:])
            df = DataFrame({'A1': np.random.default_rng(2).standard_normal(20), 'A2': np.random.default_rng(2).standard_normal(20), 'B': 'foo', 'C': 'bar'}, index=np.arange(20))
            df.loc[0:15, :] = np.nan
            _maybe_remove(store, 'df')
            store.append('df', df[:10], dropna=True)
            store.append('df', df[10:], dropna=True)
            tm.assert_frame_equal(store['df'], df, check_index_type=True)
            _maybe_remove(store, 'df2')
            store.append('df2', df[:10], dropna=False)
            store.append('df2', df[10:], dropna=False)
            tm.assert_frame_equal(store['df2'], df, check_index_type=True)
            df = DataFrame({'A1': np.random.default_rng(2).standard_normal(20), 'A2': np.random.default_rng(2).standard_normal(20), 'B': 'foo', 'C': 'bar', 'D': Timestamp('2001-01-01').as_unit('ns'), 'E': Timestamp('2001-01-02').as_unit('ns')}, index=np.arange(20))
            df.loc[0:15, :] = np.nan
            _maybe_remove(store, 'df')
            store.append('df', df[:10], dropna=True)
            store.append('df', df[10:], dropna=True)
            tm.assert_frame_equal(store['df'], df, check_index_type=True)
            _maybe_remove(store, 'df2')
            store.append('df2', df[:10], dropna=False)
            store.append('df2', df[10:], dropna=False)
            tm.assert_frame_equal(store['df2'], df, check_index_type=True)