from datetime import timedelta
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dtypes_are_correct_after_column_slice(self):
    df = DataFrame(index=range(5), columns=list('abc'), dtype=np.float64)
    tm.assert_series_equal(df.dtypes, Series({'a': np.float64, 'b': np.float64, 'c': np.float64}))
    tm.assert_series_equal(df.iloc[:, 2:].dtypes, Series({'c': np.float64}))
    tm.assert_series_equal(df.dtypes, Series({'a': np.float64, 'b': np.float64, 'c': np.float64}))