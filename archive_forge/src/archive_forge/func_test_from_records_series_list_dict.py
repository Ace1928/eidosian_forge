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
def test_from_records_series_list_dict(self):
    expected = DataFrame([[{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]]).T
    data = Series([[{'a': 1, 'b': 2}], [{'a': 3, 'b': 4}]])
    result = DataFrame.from_records(data)
    tm.assert_frame_equal(result, expected)