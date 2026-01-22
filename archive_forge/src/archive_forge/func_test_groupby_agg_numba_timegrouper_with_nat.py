from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
@pytest.mark.single_cpu
def test_groupby_agg_numba_timegrouper_with_nat(self, groupby_with_truncated_bingrouper):
    pytest.importorskip('numba')
    gb = groupby_with_truncated_bingrouper
    result = gb['Quantity'].aggregate(lambda values, index: np.nanmean(values), engine='numba')
    expected = gb['Quantity'].aggregate('mean')
    tm.assert_series_equal(result, expected)
    result_df = gb[['Quantity']].aggregate(lambda values, index: np.nanmean(values), engine='numba')
    expected_df = gb[['Quantity']].aggregate('mean')
    tm.assert_frame_equal(result_df, expected_df)