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
@pytest.mark.parametrize('names', [(None, None, None), ('foo', 'bar', None), ('baz', 'baz', 'baz')])
def test_ser_cmp_result_names(self, names, comparison_op):
    op = comparison_op
    dti = date_range('1949-06-07 03:00:00', freq='h', periods=5, name=names[0])
    ser = Series(dti).rename(names[1])
    result = op(ser, dti)
    assert result.name == names[2]
    dti = dti.tz_localize('US/Central')
    dti = pd.DatetimeIndex(dti, freq='infer')
    ser = Series(dti).rename(names[1])
    result = op(ser, dti)
    assert result.name == names[2]
    tdi = dti - dti.shift(1)
    ser = Series(tdi).rename(names[1])
    result = op(ser, tdi)
    assert result.name == names[2]
    if op in [operator.eq, operator.ne]:
        ii = pd.interval_range(start=0, periods=5, name=names[0])
        ser = Series(ii).rename(names[1])
        result = op(ser, ii)
        assert result.name == names[2]
    if op in [operator.eq, operator.ne]:
        cidx = tdi.astype('category')
        ser = Series(cidx).rename(names[1])
        result = op(ser, cidx)
        assert result.name == names[2]