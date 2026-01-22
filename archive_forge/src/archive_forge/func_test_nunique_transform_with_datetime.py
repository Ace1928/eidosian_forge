import datetime as dt
from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nunique_transform_with_datetime():
    df = DataFrame(date_range('2008-12-31', '2009-01-02'), columns=['date'])
    result = df.groupby([0, 0, 1])['date'].transform('nunique')
    expected = Series([2, 2, 1], name='date')
    tm.assert_series_equal(result, expected)