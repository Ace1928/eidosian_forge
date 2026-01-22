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
def test_append_series(setup_path):
    with ensure_clean_store(setup_path) as store:
        ss = Series(range(20), dtype=np.float64, index=[f'i_{i}' for i in range(20)])
        ts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
        ns = Series(np.arange(100))
        store.append('ss', ss)
        result = store['ss']
        tm.assert_series_equal(result, ss)
        assert result.name is None
        store.append('ts', ts)
        result = store['ts']
        tm.assert_series_equal(result, ts)
        assert result.name is None
        ns.name = 'foo'
        store.append('ns', ns)
        result = store['ns']
        tm.assert_series_equal(result, ns)
        assert result.name == ns.name
        expected = ns[ns > 60]
        result = store.select('ns', 'foo>60')
        tm.assert_series_equal(result, expected)
        expected = ns[(ns > 70) & (ns.index < 90)]
        result = store.select('ns', 'foo>70 and index<90')
        tm.assert_series_equal(result, expected, check_index_type=True)
        mi = DataFrame(np.random.default_rng(2).standard_normal((5, 1)), columns=['A'])
        mi['B'] = np.arange(len(mi))
        mi['C'] = 'foo'
        mi.loc[3:5, 'C'] = 'bar'
        mi.set_index(['C', 'B'], inplace=True)
        s = mi.stack(future_stack=True)
        s.index = s.index.droplevel(2)
        store.append('mi', s)
        tm.assert_series_equal(store['mi'], s, check_index_type=True)