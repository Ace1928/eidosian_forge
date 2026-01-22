from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_mixed3(self):
    df = DataFrame({'A': Series([3, 0], dtype='int64'), 'B': Series([0, 3], dtype='int64')})
    result = df.replace(3, df.mean().to_dict())
    expected = df.copy().astype('float64')
    m = df.mean()
    expected.iloc[0, 0] = m.iloc[0]
    expected.iloc[1, 1] = m.iloc[1]
    tm.assert_frame_equal(result, expected)