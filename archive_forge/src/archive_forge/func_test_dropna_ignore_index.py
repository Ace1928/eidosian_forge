import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val', [1, 1.5])
def test_dropna_ignore_index(self, val):
    df = DataFrame({'a': [1, 2, val]}, index=[3, 2, 1])
    result = df.dropna(ignore_index=True)
    expected = DataFrame({'a': [1, 2, val]})
    tm.assert_frame_equal(result, expected)
    df.dropna(ignore_index=True, inplace=True)
    tm.assert_frame_equal(df, expected)