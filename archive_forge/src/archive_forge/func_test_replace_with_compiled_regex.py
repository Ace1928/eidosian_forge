from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_with_compiled_regex(self):
    df = DataFrame(['a', 'b', 'c'])
    regex = re.compile('^a$')
    result = df.replace({regex: 'z'}, regex=True)
    expected = DataFrame(['z', 'b', 'c'])
    tm.assert_frame_equal(result, expected)