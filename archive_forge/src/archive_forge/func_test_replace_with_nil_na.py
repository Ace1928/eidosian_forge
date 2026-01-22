from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_with_nil_na(self):
    ser = DataFrame({'a': ['nil', pd.NA]})
    expected = DataFrame({'a': ['anything else', pd.NA]}, index=[0, 1])
    result = ser.replace('nil', 'anything else')
    tm.assert_frame_equal(expected, result)