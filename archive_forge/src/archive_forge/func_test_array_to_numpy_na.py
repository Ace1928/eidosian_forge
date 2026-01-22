import datetime
import decimal
import re
import numpy as np
import pytest
import pytz
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import register_extension_dtype
from pandas.arrays import (
from pandas.core.arrays import (
from pandas.tests.extension.decimal import (
def test_array_to_numpy_na():
    arr = pd.array([pd.NA, 1], dtype='string[python]')
    result = arr.to_numpy(na_value=True, dtype=bool)
    expected = np.array([True, True])
    tm.assert_numpy_array_equal(result, expected)