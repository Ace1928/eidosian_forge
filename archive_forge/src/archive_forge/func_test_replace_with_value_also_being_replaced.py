from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_with_value_also_being_replaced(self):
    df = DataFrame({'A': [0, 1, 2], 'B': [1, 0, 2]})
    result = df.replace({0: 1, 1: np.nan})
    expected = DataFrame({'A': [1, np.nan, 2], 'B': [np.nan, 1, 2]})
    tm.assert_frame_equal(result, expected)