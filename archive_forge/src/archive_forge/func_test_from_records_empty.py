from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_little_endian
from pandas import (
import pandas._testing as tm
def test_from_records_empty(self):
    result = DataFrame.from_records([], columns=['a', 'b', 'c'])
    expected = DataFrame(columns=['a', 'b', 'c'])
    tm.assert_frame_equal(result, expected)
    result = DataFrame.from_records([], columns=['a', 'b', 'b'])
    expected = DataFrame(columns=['a', 'b', 'b'])
    tm.assert_frame_equal(result, expected)