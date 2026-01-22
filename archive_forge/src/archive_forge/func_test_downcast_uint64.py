import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ser,expected', [(Series([0, 9223372036854775808]), Series([0, 9223372036854775808], dtype=np.uint64))])
def test_downcast_uint64(ser, expected):
    result = to_numeric(ser, downcast='unsigned')
    tm.assert_series_equal(result, expected)