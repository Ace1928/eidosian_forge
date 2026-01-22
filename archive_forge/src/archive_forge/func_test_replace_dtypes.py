from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('frame, to_replace, value, expected', [(DataFrame({'ints': [1, 2, 3]}), 1, 0, DataFrame({'ints': [0, 2, 3]})), (DataFrame({'ints': [1, 2, 3]}, dtype=np.int32), 1, 0, DataFrame({'ints': [0, 2, 3]}, dtype=np.int32)), (DataFrame({'ints': [1, 2, 3]}, dtype=np.int16), 1, 0, DataFrame({'ints': [0, 2, 3]}, dtype=np.int16)), (DataFrame({'bools': [True, False, True]}), False, True, DataFrame({'bools': [True, True, True]})), (DataFrame({'complex': [1j, 2j, 3j]}), 1j, 0, DataFrame({'complex': [0j, 2j, 3j]})), (DataFrame({'datetime64': Index([datetime(2018, 5, 28), datetime(2018, 7, 28), datetime(2018, 5, 28)])}), datetime(2018, 5, 28), datetime(2018, 7, 28), DataFrame({'datetime64': Index([datetime(2018, 7, 28)] * 3)})), (DataFrame({'dt': [datetime(3017, 12, 20)], 'str': ['foo']}), 'foo', 'bar', DataFrame({'dt': [datetime(3017, 12, 20)], 'str': ['bar']})), (DataFrame({'dt': [datetime(2920, 10, 1)]}), datetime(2920, 10, 1), datetime(2020, 10, 1), DataFrame({'dt': [datetime(2020, 10, 1)]})), (DataFrame({'A': date_range('20130101', periods=3, tz='US/Eastern'), 'B': [0, np.nan, 2]}), Timestamp('20130102', tz='US/Eastern'), Timestamp('20130104', tz='US/Eastern'), DataFrame({'A': pd.DatetimeIndex([Timestamp('20130101', tz='US/Eastern'), Timestamp('20130104', tz='US/Eastern'), Timestamp('20130103', tz='US/Eastern')]).as_unit('ns'), 'B': [0, np.nan, 2]})), (DataFrame([[1, 1.0], [2, 2.0]]), 1.0, 5, DataFrame([[5, 5.0], [2, 2.0]])), (DataFrame([[1, 1.0], [2, 2.0]]), 1, 5, DataFrame([[5, 5.0], [2, 2.0]])), (DataFrame([[1, 1.0], [2, 2.0]]), 1.0, 5.0, DataFrame([[5, 5.0], [2, 2.0]])), (DataFrame([[1, 1.0], [2, 2.0]]), 1, 5.0, DataFrame([[5, 5.0], [2, 2.0]]))])
def test_replace_dtypes(self, frame, to_replace, value, expected):
    warn = None
    if isinstance(to_replace, datetime) and to_replace.year == 2920:
        warn = FutureWarning
    msg = 'Downcasting behavior in `replace` '
    with tm.assert_produces_warning(warn, match=msg):
        result = frame.replace(to_replace, value)
    tm.assert_frame_equal(result, expected)