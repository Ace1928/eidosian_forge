import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_api_2(tmp_path, setup_path):
    path = tmp_path / setup_path
    df = DataFrame(range(20))
    df.to_hdf(path, key='df', append=False, format='fixed')
    tm.assert_frame_equal(read_hdf(path, 'df'), df)
    df.to_hdf(path, key='df', append=False, format='f')
    tm.assert_frame_equal(read_hdf(path, 'df'), df)
    df.to_hdf(path, key='df', append=False)
    tm.assert_frame_equal(read_hdf(path, 'df'), df)
    df.to_hdf(path, key='df')
    tm.assert_frame_equal(read_hdf(path, 'df'), df)
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(range(20))
        _maybe_remove(store, 'df')
        store.append('df', df.iloc[:10], append=True, format='table')
        store.append('df', df.iloc[10:], append=True, format='table')
        tm.assert_frame_equal(store.select('df'), df)
        _maybe_remove(store, 'df')
        store.append('df', df.iloc[:10], append=False, format='table')
        store.append('df', df.iloc[10:], append=True, format='table')
        tm.assert_frame_equal(store.select('df'), df)
        _maybe_remove(store, 'df')
        store.append('df', df.iloc[:10], append=False, format='table')
        store.append('df', df.iloc[10:], append=True, format='table')
        tm.assert_frame_equal(store.select('df'), df)
        _maybe_remove(store, 'df')
        store.append('df', df.iloc[:10], append=False, format='table')
        store.append('df', df.iloc[10:], append=True, format=None)
        tm.assert_frame_equal(store.select('df'), df)