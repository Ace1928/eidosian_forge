import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby_multilevel(self, multiindex_year_month_day_dataframe_random_data):
    ymd = multiindex_year_month_day_dataframe_random_data
    result = ymd.groupby(level=[0, 1]).mean()
    k1 = ymd.index.get_level_values(0)
    k2 = ymd.index.get_level_values(1)
    expected = ymd.groupby([k1, k2]).mean()
    tm.assert_frame_equal(result, expected, check_names=False)
    assert result.index.names == ymd.index.names[:2]
    result2 = ymd.groupby(level=ymd.index.names[:2]).mean()
    tm.assert_frame_equal(result, result2)