from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
def test_series_constructor_coerce_data_to_extension_dtype():
    dtype = DecimalDtype()
    ser = pd.Series([0, 1, 2], dtype=dtype)
    arr = DecimalArray([decimal.Decimal(0), decimal.Decimal(1), decimal.Decimal(2)], dtype=dtype)
    exp = pd.Series(arr)
    tm.assert_series_equal(ser, exp)