from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('name', [None, 'foo', 2, 3.0, pd.Timedelta(6), Timestamp('2012-12-30', tz='UTC'), '2012-12-31'])
def test_reset_index_with_datetimeindex_cols(self, name):
    df = DataFrame([[1, 2], [3, 4]], columns=date_range('1/1/2013', '1/2/2013'), index=['A', 'B'])
    df.index.name = name
    result = df.reset_index()
    item = name if name is not None else 'index'
    columns = Index([item, datetime(2013, 1, 1), datetime(2013, 1, 2)])
    if isinstance(item, str) and item == '2012-12-31':
        columns = columns.astype('datetime64[ns]')
    else:
        assert columns.dtype == object
    expected = DataFrame([['A', 1, 2], ['B', 3, 4]], columns=columns)
    tm.assert_frame_equal(result, expected)