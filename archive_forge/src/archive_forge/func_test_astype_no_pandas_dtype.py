from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_astype_no_pandas_dtype(self):
    ser = Series([1, 2], dtype='int64')
    result = ser.astype(ser.array.dtype)
    tm.assert_series_equal(result, ser)