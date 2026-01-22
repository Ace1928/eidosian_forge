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
def test_mean_extensionarray_numeric_only_true(self):
    arr = np.random.default_rng(2).integers(1000, size=(10, 5))
    df = DataFrame(arr, dtype='Int64')
    result = df.mean(numeric_only=True)
    expected = DataFrame(arr).mean().astype('Float64')
    tm.assert_series_equal(result, expected)