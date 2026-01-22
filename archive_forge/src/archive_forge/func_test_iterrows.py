import datetime
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_iterrows(self, float_frame, float_string_frame):
    for k, v in float_frame.iterrows():
        exp = float_frame.loc[k]
        tm.assert_series_equal(v, exp)
    for k, v in float_string_frame.iterrows():
        exp = float_string_frame.loc[k]
        tm.assert_series_equal(v, exp)