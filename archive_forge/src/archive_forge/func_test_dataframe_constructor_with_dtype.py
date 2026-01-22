from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
def test_dataframe_constructor_with_dtype():
    arr = DecimalArray([decimal.Decimal('10.0')])
    result = pd.DataFrame({'A': arr}, dtype=DecimalDtype())
    expected = pd.DataFrame({'A': arr})
    tm.assert_frame_equal(result, expected)
    arr = DecimalArray([decimal.Decimal('10.0')])
    result = pd.DataFrame({'A': arr}, dtype='int64')
    expected = pd.DataFrame({'A': [10]})
    tm.assert_frame_equal(result, expected)