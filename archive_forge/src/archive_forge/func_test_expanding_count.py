import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_expanding_count(series):
    result = series.expanding(min_periods=0).count()
    tm.assert_almost_equal(result, series.rolling(window=len(series), min_periods=0).count())