from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_groupby_apply_timegrouper_with_nat_apply_squeeze(self, frame_for_truncated_bingrouper):
    df = frame_for_truncated_bingrouper
    tdg = Grouper(key='Date', freq='100YE')
    gb = df.groupby(tdg)
    assert gb.ngroups == 1
    assert gb._selected_obj._get_axis(gb.axis).nlevels == 1
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        res = gb.apply(lambda x: x['Quantity'] * 2)
    dti = Index([Timestamp('2013-12-31')], dtype=df['Date'].dtype, name='Date')
    expected = DataFrame([[36, 6, 6, 10, 2]], index=dti, columns=Index([0, 1, 5, 2, 3], name='Quantity'))
    tm.assert_frame_equal(res, expected)