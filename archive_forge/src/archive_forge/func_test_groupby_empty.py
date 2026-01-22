from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_empty(self):
    s = Series([], name='name', dtype='float64')
    gr = s.groupby([])
    result = gr.mean()
    expected = s.set_axis(Index([], dtype=np.intp))
    tm.assert_series_equal(result, expected)
    assert len(gr._grouper.groupings) == 1
    tm.assert_numpy_array_equal(gr._grouper.group_info[0], np.array([], dtype=np.dtype(np.intp)))
    tm.assert_numpy_array_equal(gr._grouper.group_info[1], np.array([], dtype=np.dtype(np.intp)))
    assert gr._grouper.group_info[2] == 0
    gb = s.groupby(s)
    msg = 'SeriesGroupBy.grouper is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouper = gb.grouper
    result = grouper.names
    expected = ['name']
    assert result == expected