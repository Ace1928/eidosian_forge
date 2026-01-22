from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_idxminmax_object_decimals(self):
    df = DataFrame({'idx': [0, 1], 'x': [Decimal('8.68'), Decimal('42.23')], 'y': [Decimal('7.11'), Decimal('79.61')]})
    res = df.idxmax()
    exp = Series({'idx': 1, 'x': 1, 'y': 1})
    tm.assert_series_equal(res, exp)
    res2 = df.idxmin()
    exp2 = exp - 1
    tm.assert_series_equal(res2, exp2)