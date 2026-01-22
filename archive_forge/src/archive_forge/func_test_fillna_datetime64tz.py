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
@pytest.mark.parametrize('fill_val,fill_dtype', [(pd.Timestamp('2012-01-01', tz='US/Eastern'), 'datetime64[ns, US/Eastern]'), (pd.Timestamp('2012-01-01'), object), (pd.Timestamp('2012-01-01', tz='Asia/Tokyo'), 'datetime64[ns, US/Eastern]'), (1, object), ('x', object)])
def test_fillna_datetime64tz(self, index_or_series, fill_val, fill_dtype):
    klass = index_or_series
    tz = 'US/Eastern'
    obj = klass([pd.Timestamp('2011-01-01', tz=tz), pd.NaT, pd.Timestamp('2011-01-03', tz=tz), pd.Timestamp('2011-01-04', tz=tz)])
    assert obj.dtype == 'datetime64[ns, US/Eastern]'
    if getattr(fill_val, 'tz', None) is None:
        fv = fill_val
    else:
        fv = fill_val.tz_convert(tz)
    exp = klass([pd.Timestamp('2011-01-01', tz=tz), fv, pd.Timestamp('2011-01-03', tz=tz), pd.Timestamp('2011-01-04', tz=tz)])
    self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)