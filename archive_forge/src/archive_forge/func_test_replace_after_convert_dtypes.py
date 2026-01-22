from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_after_convert_dtypes(self):
    df = DataFrame({'grp': [1, 2, 3, 4, 5]}, dtype='Int64')
    result = df.replace(1, 10)
    expected = DataFrame({'grp': [10, 2, 3, 4, 5]}, dtype='Int64')
    tm.assert_frame_equal(result, expected)