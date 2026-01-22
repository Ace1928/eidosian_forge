from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_value_is_none(self, datetime_frame):
    orig_value = datetime_frame.iloc[0, 0]
    orig2 = datetime_frame.iloc[1, 0]
    datetime_frame.iloc[0, 0] = np.nan
    datetime_frame.iloc[1, 0] = 1
    result = datetime_frame.replace(to_replace={np.nan: 0})
    expected = datetime_frame.T.replace(to_replace={np.nan: 0}).T
    tm.assert_frame_equal(result, expected)
    result = datetime_frame.replace(to_replace={np.nan: 0, 1: -100000000.0})
    tsframe = datetime_frame.copy()
    tsframe.iloc[0, 0] = 0
    tsframe.iloc[1, 0] = -100000000.0
    expected = tsframe
    tm.assert_frame_equal(expected, result)
    datetime_frame.iloc[0, 0] = orig_value
    datetime_frame.iloc[1, 0] = orig2