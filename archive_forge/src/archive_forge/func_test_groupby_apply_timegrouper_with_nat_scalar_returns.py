from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_groupby_apply_timegrouper_with_nat_scalar_returns(self, groupby_with_truncated_bingrouper):
    gb = groupby_with_truncated_bingrouper
    res = gb['Quantity'].apply(lambda x: x.iloc[0] if len(x) else np.nan)
    df = gb.obj
    unit = df['Date']._values.unit
    dti = date_range('2013-09-01', '2013-10-01', freq='5D', name='Date', unit=unit)
    expected = Series([18, np.nan, np.nan, np.nan, np.nan, np.nan, 5], index=dti._with_freq(None), name='Quantity')
    tm.assert_series_equal(res, expected)