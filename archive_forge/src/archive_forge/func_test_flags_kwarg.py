from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_flags_kwarg(any_string_dtype):
    data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com', 'Rob': 'rob@gmail.com', 'Wes': np.nan}
    data = Series(data, dtype=any_string_dtype)
    pat = '([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\\.([A-Z]{2,4})'
    use_pyarrow = using_pyarrow(any_string_dtype)
    result = data.str.extract(pat, flags=re.IGNORECASE, expand=True)
    assert result.iloc[0].tolist() == ['dave', 'google', 'com']
    with tm.maybe_produces_warning(PerformanceWarning, use_pyarrow):
        result = data.str.match(pat, flags=re.IGNORECASE)
    assert result.iloc[0]
    with tm.maybe_produces_warning(PerformanceWarning, use_pyarrow):
        result = data.str.fullmatch(pat, flags=re.IGNORECASE)
    assert result.iloc[0]
    result = data.str.findall(pat, flags=re.IGNORECASE)
    assert result.iloc[0][0] == ('dave', 'google', 'com')
    result = data.str.count(pat, flags=re.IGNORECASE)
    assert result.iloc[0] == 1
    msg = 'has match groups'
    with tm.assert_produces_warning(UserWarning, match=msg, raise_on_extra_warnings=not use_pyarrow):
        result = data.str.contains(pat, flags=re.IGNORECASE)
    assert result.iloc[0]