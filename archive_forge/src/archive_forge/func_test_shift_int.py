import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_int(self, datetime_frame, frame_or_series):
    ts = tm.get_obj(datetime_frame, frame_or_series).astype(int)
    shifted = ts.shift(1)
    expected = ts.astype(float).shift(1)
    tm.assert_equal(shifted, expected)