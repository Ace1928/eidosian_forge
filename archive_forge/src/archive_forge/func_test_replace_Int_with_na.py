import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_Int_with_na(self, any_int_ea_dtype):
    result = pd.Series([0, None], dtype=any_int_ea_dtype).replace(0, pd.NA)
    expected = pd.Series([pd.NA, pd.NA], dtype=any_int_ea_dtype)
    tm.assert_series_equal(result, expected)
    result = pd.Series([0, 1], dtype=any_int_ea_dtype).replace(0, pd.NA)
    result.replace(1, pd.NA, inplace=True)
    tm.assert_series_equal(result, expected)