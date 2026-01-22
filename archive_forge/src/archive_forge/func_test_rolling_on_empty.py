import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_rolling_on_empty(self):
    df = DataFrame({'column': []}, index=[])
    result = df.rolling('5s').min()
    expected = DataFrame({'column': []}, index=[])
    tm.assert_frame_equal(result, expected)