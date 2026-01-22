import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_diff_period(self):
    pi = date_range('2016-01-01', periods=3).to_period('D')
    df = DataFrame({'A': pi})
    result = df.diff(1, axis=1)
    expected = (df - pd.NaT).astype(object)
    tm.assert_frame_equal(result, expected)