import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_no_by_inplace(self):
    df = DataFrame({'a': [1, 2, 3]})
    expected = df.copy()
    result = df.sort_values(by=[], inplace=True)
    tm.assert_frame_equal(df, expected)
    assert result is None