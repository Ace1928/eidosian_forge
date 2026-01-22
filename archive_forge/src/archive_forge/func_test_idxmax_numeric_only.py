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
@pytest.mark.parametrize('numeric_only', [True, False])
def test_idxmax_numeric_only(self, numeric_only):
    df = DataFrame({'a': [2, 3, 1], 'b': [2, 1, 1], 'c': list('xyx')})
    result = df.idxmax(numeric_only=numeric_only)
    if numeric_only:
        expected = Series([1, 0], index=['a', 'b'])
    else:
        expected = Series([1, 0, 1], index=['a', 'b', 'c'])
    tm.assert_series_equal(result, expected)