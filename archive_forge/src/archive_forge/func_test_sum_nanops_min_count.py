from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_sum_nanops_min_count(self):
    df = DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    result = df.sum(min_count=10)
    expected = Series([np.nan, np.nan], index=['x', 'y'])
    tm.assert_series_equal(result, expected)