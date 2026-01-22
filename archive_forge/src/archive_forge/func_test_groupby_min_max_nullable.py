import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['Int64', 'Int32', 'Float64', 'Float32', 'boolean'])
def test_groupby_min_max_nullable(dtype):
    if dtype == 'Int64':
        ts = 1618556707013635762
    elif dtype == 'boolean':
        ts = 0
    else:
        ts = 4.0
    df = DataFrame({'id': [2, 2], 'ts': [ts, ts + 1]})
    df['ts'] = df['ts'].astype(dtype)
    gb = df.groupby('id')
    result = gb.min()
    expected = df.iloc[:1].set_index('id')
    tm.assert_frame_equal(result, expected)
    res_max = gb.max()
    expected_max = df.iloc[1:].set_index('id')
    tm.assert_frame_equal(res_max, expected_max)
    result2 = gb.min(min_count=3)
    expected2 = DataFrame({'ts': [pd.NA]}, index=expected.index, dtype=dtype)
    tm.assert_frame_equal(result2, expected2)
    res_max2 = gb.max(min_count=3)
    tm.assert_frame_equal(res_max2, expected2)
    df2 = DataFrame({'id': [2, 2, 2], 'ts': [ts, pd.NA, ts + 1]})
    df2['ts'] = df2['ts'].astype(dtype)
    gb2 = df2.groupby('id')
    result3 = gb2.min()
    tm.assert_frame_equal(result3, expected)
    res_max3 = gb2.max()
    tm.assert_frame_equal(res_max3, expected_max)
    result4 = gb2.min(min_count=100)
    tm.assert_frame_equal(result4, expected2)
    res_max4 = gb2.max(min_count=100)
    tm.assert_frame_equal(res_max4, expected2)