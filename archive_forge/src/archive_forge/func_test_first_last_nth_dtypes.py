import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_first_last_nth_dtypes(df_mixed_floats):
    df = df_mixed_floats.copy()
    df['E'] = True
    df['F'] = 1
    grouped = df.groupby('A')
    first = grouped.first()
    expected = df.loc[[1, 0], ['B', 'C', 'D', 'E', 'F']]
    expected.index = Index(['bar', 'foo'], name='A')
    expected = expected.sort_index()
    tm.assert_frame_equal(first, expected)
    last = grouped.last()
    expected = df.loc[[5, 7], ['B', 'C', 'D', 'E', 'F']]
    expected.index = Index(['bar', 'foo'], name='A')
    expected = expected.sort_index()
    tm.assert_frame_equal(last, expected)
    nth = grouped.nth(1)
    expected = df.iloc[[2, 3]]
    tm.assert_frame_equal(nth, expected)
    idx = list(range(10))
    idx.append(9)
    s = Series(data=range(11), index=idx, name='IntCol')
    assert s.dtype == 'int64'
    f = s.groupby(level=0).first()
    assert f.dtype == 'int64'