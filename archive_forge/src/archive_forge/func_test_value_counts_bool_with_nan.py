import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ser, dropna, exp', [(Series([False, True, True, pd.NA]), False, Series([2, 1, 1], index=[True, False, pd.NA], name='count')), (Series([False, True, True, pd.NA]), True, Series([2, 1], index=Index([True, False], dtype=object), name='count')), (Series(range(3), index=[True, False, np.nan]).index, False, Series([1, 1, 1], index=[True, False, np.nan], name='count'))])
def test_value_counts_bool_with_nan(self, ser, dropna, exp):
    out = ser.value_counts(dropna=dropna)
    tm.assert_series_equal(out, exp)