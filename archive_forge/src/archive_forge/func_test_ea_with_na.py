import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_ea_with_na(self, any_numeric_ea_dtype):
    df = DataFrame({'a': [1, pd.NA, pd.NA], 'b': pd.NA}, dtype=any_numeric_ea_dtype)
    result = df.describe()
    expected = DataFrame({'a': [1.0, 1.0, pd.NA] + [1.0] * 5, 'b': [0.0] + [pd.NA] * 7}, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], dtype='Float64')
    tm.assert_frame_equal(result, expected)