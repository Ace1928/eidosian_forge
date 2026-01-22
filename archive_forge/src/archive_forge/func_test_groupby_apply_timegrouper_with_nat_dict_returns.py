from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_groupby_apply_timegrouper_with_nat_dict_returns(self, groupby_with_truncated_bingrouper):
    gb = groupby_with_truncated_bingrouper
    res = gb['Quantity'].apply(lambda x: {'foo': len(x)})
    df = gb.obj
    unit = df['Date']._values.unit
    dti = date_range('2013-09-01', '2013-10-01', freq='5D', name='Date', unit=unit)
    mi = MultiIndex.from_arrays([dti, ['foo'] * len(dti)])
    expected = Series([3, 0, 0, 0, 0, 0, 2], index=mi, name='Quantity')
    tm.assert_series_equal(res, expected)