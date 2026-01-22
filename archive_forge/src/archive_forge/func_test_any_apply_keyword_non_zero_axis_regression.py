from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_any_apply_keyword_non_zero_axis_regression():
    df = DataFrame({'A': [1, 2, 0], 'B': [0, 2, 0], 'C': [0, 0, 0]})
    expected = Series([True, True, False])
    tm.assert_series_equal(df.any(axis=1), expected)
    result = df.apply('any', axis=1)
    tm.assert_series_equal(result, expected)
    result = df.apply('any', 1)
    tm.assert_series_equal(result, expected)