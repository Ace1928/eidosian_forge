import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_iter_box_dt64(self, unit):
    vals = [Timestamp('2011-01-01'), Timestamp('2011-01-02')]
    ser = Series(vals).dt.as_unit(unit)
    assert ser.dtype == f'datetime64[{unit}]'
    for res, exp in zip(ser, vals):
        assert isinstance(res, Timestamp)
        assert res.tz is None
        assert res == exp
        assert res.unit == unit