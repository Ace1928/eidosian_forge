from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_with_Time_Grouper(unit):
    idx2 = to_datetime(['2016-08-31 22:08:12.000', '2016-08-31 22:09:12.200', '2016-08-31 22:20:12.400']).as_unit(unit)
    test_data = DataFrame({'quant': [1.0, 1.0, 3.0], 'quant2': [1.0, 1.0, 3.0], 'time2': idx2})
    time2 = date_range('2016-08-31 22:08:00', periods=13, freq='1min', unit=unit)
    expected_output = DataFrame({'time2': time2, 'quant': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'quant2': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]})
    gb = test_data.groupby(Grouper(key='time2', freq='1min'))
    result = gb.count().reset_index()
    tm.assert_frame_equal(result, expected_output)