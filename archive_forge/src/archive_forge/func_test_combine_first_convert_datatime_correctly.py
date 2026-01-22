from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data1, data2, data_expected', (([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], [pd.NaT, pd.NaT, pd.NaT], [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]), ([pd.NaT, pd.NaT, pd.NaT], [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]), ([datetime(2000, 1, 2), pd.NaT, pd.NaT], [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], [datetime(2000, 1, 2), datetime(2000, 1, 2), datetime(2000, 1, 3)]), ([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], [datetime(2000, 1, 2), pd.NaT, pd.NaT], [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)])))
def test_combine_first_convert_datatime_correctly(self, data1, data2, data_expected):
    df1, df2 = (DataFrame({'a': data1}), DataFrame({'a': data2}))
    result = df1.combine_first(df2)
    expected = DataFrame({'a': data_expected})
    tm.assert_frame_equal(result, expected)