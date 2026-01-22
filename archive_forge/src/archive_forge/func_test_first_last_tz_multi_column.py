import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method, ts, alpha', [['first', Timestamp('2013-01-01', tz='US/Eastern'), 'a'], ['last', Timestamp('2013-01-02', tz='US/Eastern'), 'b']])
def test_first_last_tz_multi_column(method, ts, alpha):
    category_string = Series(list('abc')).astype('category')
    df = DataFrame({'group': [1, 1, 2], 'category_string': category_string, 'datetimetz': pd.date_range('20130101', periods=3, tz='US/Eastern')})
    result = getattr(df.groupby('group'), method)()
    expected = DataFrame({'category_string': pd.Categorical([alpha, 'c'], dtype=category_string.dtype), 'datetimetz': [ts, Timestamp('2013-01-03', tz='US/Eastern')]}, index=Index([1, 2], name='group'))
    tm.assert_frame_equal(result, expected)