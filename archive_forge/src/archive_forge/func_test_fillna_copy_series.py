import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort
def test_fillna_copy_series(self, data_missing):
    arr = data_missing.take([1, 1])
    ser = pd.Series(arr, copy=False)
    ser_orig = ser.copy()
    filled_val = ser[0]
    result = ser.fillna(filled_val)
    result.iloc[0] = filled_val
    tm.assert_series_equal(ser, ser_orig)