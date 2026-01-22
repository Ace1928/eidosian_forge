from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['pad', 'bfill'])
@pytest.mark.parametrize('axis', [0, 1, None])
@pytest.mark.parametrize('fill_axis', [0, 1])
@pytest.mark.parametrize('how', ['inner', 'outer', 'left', 'right'])
@pytest.mark.parametrize('left_slice', [[slice(4), slice(10)], [slice(0), slice(0)]])
@pytest.mark.parametrize('right_slice', [[slice(2, None), slice(6, None)], [slice(0), slice(0)]])
@pytest.mark.parametrize('limit', [1, None])
def test_align_fill_method(self, how, method, axis, fill_axis, float_frame, left_slice, right_slice, limit):
    frame = float_frame
    left = frame.iloc[left_slice[0], left_slice[1]]
    right = frame.iloc[right_slice[0], right_slice[1]]
    msg = "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align are deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        aa, ab = left.align(right, axis=axis, join=how, method=method, limit=limit, fill_axis=fill_axis)
    join_index, join_columns = (None, None)
    ea, eb = (left, right)
    if axis is None or axis == 0:
        join_index = left.index.join(right.index, how=how)
        ea = ea.reindex(index=join_index)
        eb = eb.reindex(index=join_index)
    if axis is None or axis == 1:
        join_columns = left.columns.join(right.columns, how=how)
        ea = ea.reindex(columns=join_columns)
        eb = eb.reindex(columns=join_columns)
    msg = "DataFrame.fillna with 'method' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        ea = ea.fillna(axis=fill_axis, method=method, limit=limit)
        eb = eb.fillna(axis=fill_axis, method=method, limit=limit)
    tm.assert_frame_equal(aa, ea)
    tm.assert_frame_equal(ab, eb)