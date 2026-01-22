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
@pytest.mark.parametrize('fill_val,exp_dtype', [(pd.Timestamp('2012-01-01'), 'datetime64[ns]'), (pd.Timestamp('2012-01-01', tz='US/Eastern'), object)], ids=['datetime64', 'datetime64tz'])
def test_where_datetime64(self, index_or_series, fill_val, exp_dtype):
    klass = index_or_series
    obj = klass(pd.date_range('2011-01-01', periods=4, freq='D')._with_freq(None))
    assert obj.dtype == 'datetime64[ns]'
    fv = fill_val
    if exp_dtype == 'datetime64[ns]':
        for scalar in [fv, fv.to_pydatetime(), fv.to_datetime64()]:
            self._run_test(obj, scalar, klass, exp_dtype)
    else:
        for scalar in [fv, fv.to_pydatetime()]:
            self._run_test(obj, fill_val, klass, exp_dtype)