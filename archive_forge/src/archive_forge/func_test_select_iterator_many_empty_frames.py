import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
def test_select_iterator_many_empty_frames(setup_path):
    chunksize = 10000
    with ensure_clean_store(setup_path) as store:
        expected = DataFrame(np.random.default_rng(2).standard_normal((100064, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=100064, freq='s'))
        _maybe_remove(store, 'df')
        store.append('df', expected)
        beg_dt = expected.index[0]
        end_dt = expected.index[chunksize - 1]
        where = f"index >= '{beg_dt}'"
        results = list(store.select('df', where=where, chunksize=chunksize))
        result = concat(results)
        rexpected = expected[expected.index >= beg_dt]
        tm.assert_frame_equal(rexpected, result)
        where = f"index <= '{end_dt}'"
        results = list(store.select('df', where=where, chunksize=chunksize))
        assert len(results) == 1
        result = concat(results)
        rexpected = expected[expected.index <= end_dt]
        tm.assert_frame_equal(rexpected, result)
        where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
        results = list(store.select('df', where=where, chunksize=chunksize))
        assert len(results) == 1
        result = concat(results)
        rexpected = expected[(expected.index >= beg_dt) & (expected.index <= end_dt)]
        tm.assert_frame_equal(rexpected, result)
        where = f"index <= '{beg_dt}' & index >= '{end_dt}'"
        results = list(store.select('df', where=where, chunksize=chunksize))
        assert len(results) == 0