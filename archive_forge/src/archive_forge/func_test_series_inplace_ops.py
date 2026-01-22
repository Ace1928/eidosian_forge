from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
@pytest.mark.parametrize('dtype1, dtype2, dtype_expected, dtype_mul', (('Int64', 'Int64', 'Int64', 'Int64'), ('float', 'float', 'float', 'float'), ('Int64', 'float', 'Float64', 'Float64'), ('Int64', 'Float64', 'Float64', 'Float64')))
def test_series_inplace_ops(self, dtype1, dtype2, dtype_expected, dtype_mul):
    ser1 = Series([1], dtype=dtype1)
    ser2 = Series([2], dtype=dtype2)
    ser1 += ser2
    expected = Series([3], dtype=dtype_expected)
    tm.assert_series_equal(ser1, expected)
    ser1 -= ser2
    expected = Series([1], dtype=dtype_expected)
    tm.assert_series_equal(ser1, expected)
    ser1 *= ser2
    expected = Series([2], dtype=dtype_mul)
    tm.assert_series_equal(ser1, expected)