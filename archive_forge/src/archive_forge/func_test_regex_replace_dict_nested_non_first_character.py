from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_regex_replace_dict_nested_non_first_character(self, any_string_dtype, using_infer_string):
    dtype = any_string_dtype
    df = DataFrame({'first': ['abc', 'bca', 'cab']}, dtype=dtype)
    if using_infer_string and any_string_dtype == 'object':
        with tm.assert_produces_warning(FutureWarning, match='Downcasting'):
            result = df.replace({'a': '.'}, regex=True)
        expected = DataFrame({'first': ['.bc', 'bc.', 'c.b']})
    else:
        result = df.replace({'a': '.'}, regex=True)
        expected = DataFrame({'first': ['.bc', 'bc.', 'c.b']}, dtype=dtype)
    tm.assert_frame_equal(result, expected)