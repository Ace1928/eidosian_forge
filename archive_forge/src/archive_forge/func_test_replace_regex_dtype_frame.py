from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't set float into string")
@pytest.mark.parametrize('regex', [False, True])
def test_replace_regex_dtype_frame(self, regex):
    df1 = DataFrame({'A': ['0'], 'B': ['0']})
    expected_df1 = DataFrame({'A': [1], 'B': [1]})
    msg = 'Downcasting behavior in `replace`'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result_df1 = df1.replace(to_replace='0', value=1, regex=regex)
    tm.assert_frame_equal(result_df1, expected_df1)
    df2 = DataFrame({'A': ['0'], 'B': ['1']})
    expected_df2 = DataFrame({'A': [1], 'B': ['1']})
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result_df2 = df2.replace(to_replace='0', value=1, regex=regex)
    tm.assert_frame_equal(result_df2, expected_df2)