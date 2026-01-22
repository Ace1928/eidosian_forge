import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_put_mixed_type(setup_path):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    df['obj1'] = 'foo'
    df['obj2'] = 'bar'
    df['bool1'] = df['A'] > 0
    df['bool2'] = df['B'] > 0
    df['bool3'] = True
    df['int1'] = 1
    df['int2'] = 2
    df['timestamp1'] = Timestamp('20010102').as_unit('ns')
    df['timestamp2'] = Timestamp('20010103').as_unit('ns')
    df['datetime1'] = Timestamp('20010102').as_unit('ns')
    df['datetime2'] = Timestamp('20010103').as_unit('ns')
    df.loc[df.index[3:6], ['obj1']] = np.nan
    df = df._consolidate()
    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, 'df')
        with tm.assert_produces_warning(pd.errors.PerformanceWarning):
            store.put('df', df)
        expected = store.get('df')
        tm.assert_frame_equal(expected, df)