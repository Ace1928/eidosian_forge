from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('other', [Timedelta(hours=31), Timedelta(hours=31).to_pytimedelta(), Timedelta(hours=31).to_timedelta64(), Timedelta(hours=31).to_timedelta64().astype('m8[h]'), np.timedelta64('NaT'), np.timedelta64('NaT', 'D'), pd.offsets.Minute(3), pd.offsets.Second(0), pd.Timestamp('2021-01-01', tz='Asia/Tokyo'), pd.Timestamp('2021-01-01'), pd.Timestamp('2021-01-01').to_pydatetime(), pd.Timestamp('2021-01-01', tz='UTC').to_pydatetime(), pd.Timestamp('2021-01-01').to_datetime64(), np.datetime64('NaT', 'ns'), pd.NaT], ids=repr)
def test_add_sub_datetimedeltalike_invalid(self, numeric_idx, other, box_with_array):
    box = box_with_array
    left = tm.box_expected(numeric_idx, box)
    msg = '|'.join(['unsupported operand type', 'Addition/subtraction of integers and integer-arrays', 'Instead of adding/subtracting', 'cannot use operands with types dtype', 'Concatenation operation is not implemented for NumPy arrays', 'Cannot (add|subtract) NaT (to|from) ndarray', 'operand type\\(s\\) all returned NotImplemented from __array_ufunc__', 'can only perform ops with numeric values', 'cannot subtract DatetimeArray from ndarray', 'Cannot add or subtract Timedelta from integers'])
    assert_invalid_addsub_type(left, other, msg)