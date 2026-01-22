from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_to_records_with_na_record(self):
    df = DataFrame([['a', 'b'], [np.nan, np.nan], ['e', 'f']], columns=[np.nan, 'right'])
    df['record'] = df[[np.nan, 'right']].to_records()
    expected = '   NaN right         record\n0    a     b      [0, a, b]\n1  NaN   NaN  [1, nan, nan]\n2    e     f      [2, e, f]'
    result = repr(df)
    assert result == expected