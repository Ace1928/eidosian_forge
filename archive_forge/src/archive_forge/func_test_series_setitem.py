import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_setitem(self, multiindex_year_month_day_dataframe_random_data, warn_copy_on_write):
    ymd = multiindex_year_month_day_dataframe_random_data
    s = ymd['A']
    with tm.assert_cow_warning(warn_copy_on_write):
        s[2000, 3] = np.nan
    assert isna(s.values[42:65]).all()
    assert notna(s.values[:42]).all()
    assert notna(s.values[65:]).all()
    with tm.assert_cow_warning(warn_copy_on_write):
        s[2000, 3, 10] = np.nan
    assert isna(s.iloc[49])
    with pytest.raises(KeyError, match='49'):
        s[49]