import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_datetime_is_numeric_includes_datetime(self):
    df = DataFrame({'a': date_range('2012', periods=3), 'b': [1, 2, 3]})
    result = df.describe()
    expected = DataFrame({'a': [3, Timestamp('2012-01-02'), Timestamp('2012-01-01'), Timestamp('2012-01-01T12:00:00'), Timestamp('2012-01-02'), Timestamp('2012-01-02T12:00:00'), Timestamp('2012-01-03'), np.nan], 'b': [3, 2, 1, 1.5, 2, 2.5, 3, 1]}, index=['count', 'mean', 'min', '25%', '50%', '75%', 'max', 'std'])
    tm.assert_frame_equal(result, expected)