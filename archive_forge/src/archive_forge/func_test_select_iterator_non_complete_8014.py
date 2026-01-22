import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
def test_select_iterator_non_complete_8014(setup_path):
    chunksize = 10000.0
    with ensure_clean_store(setup_path) as store:
        expected = DataFrame(np.random.default_rng(2).standard_normal((100064, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=100064, freq='s'))
        _maybe_remove(store, 'df')
        store.append('df', expected)
        beg_dt = expected.index[1]
        end_dt = expected.index[-2]
        where = f"index >= '{beg_dt}'"
        results = list(store.select('df', where=where, chunksize=chunksize))
        result = concat(results)
        rexpected = expected[expected.index >= beg_dt]
        tm.assert_frame_equal(rexpected, result)
        where = f"index <= '{end_dt}'"
        results = list(store.select('df', where=where, chunksize=chunksize))
        result = concat(results)
        rexpected = expected[expected.index <= end_dt]
        tm.assert_frame_equal(rexpected, result)
        where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
        results = list(store.select('df', where=where, chunksize=chunksize))
        result = concat(results)
        rexpected = expected[(expected.index >= beg_dt) & (expected.index <= end_dt)]
        tm.assert_frame_equal(rexpected, result)
    with ensure_clean_store(setup_path) as store:
        expected = DataFrame(np.random.default_rng(2).standard_normal((100064, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=100064, freq='s'))
        _maybe_remove(store, 'df')
        store.append('df', expected)
        end_dt = expected.index[-1]
        where = f"index > '{end_dt}'"
        results = list(store.select('df', where=where, chunksize=chunksize))
        assert 0 == len(results)