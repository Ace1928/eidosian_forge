import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,exp_data', [([200, 300, '', 'NaN', 30000000000000000000], [200, 300, np.nan, np.nan, 30000000000000000000]), (['12345678901234567890', '1234567890', 'ITEM'], [12345678901234567890, 1234567890, np.nan])])
def test_coerce_uint64_conflict(data, exp_data):
    result = to_numeric(Series(data), errors='coerce')
    expected = Series(exp_data, dtype=float)
    tm.assert_series_equal(result, expected)