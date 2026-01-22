from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
def test_setitem_series_no_coercion_from_values_list(self):
    ser = pd.Series(['a', 1])
    ser[:] = list(ser.values)
    expected = pd.Series(['a', 1])
    tm.assert_series_equal(ser, expected)