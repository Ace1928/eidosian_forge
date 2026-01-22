from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_with_datetime_tz(self):
    dr = date_range('20130101', periods=3, tz='US/Eastern')
    s = Series(dr)
    assert s.dtype.name == 'datetime64[ns, US/Eastern]'
    assert s.dtype == 'datetime64[ns, US/Eastern]'
    assert isinstance(s.dtype, DatetimeTZDtype)
    assert 'datetime64[ns, US/Eastern]' in str(s)
    result = s.values
    assert isinstance(result, np.ndarray)
    assert result.dtype == 'datetime64[ns]'
    exp = DatetimeIndex(result)
    exp = exp.tz_localize('UTC').tz_convert(tz=s.dt.tz)
    tm.assert_index_equal(dr, exp)
    result = s.iloc[0]
    assert result == Timestamp('2013-01-01 00:00:00-0500', tz='US/Eastern')
    result = s[0]
    assert result == Timestamp('2013-01-01 00:00:00-0500', tz='US/Eastern')
    result = s[Series([True, True, False], index=s.index)]
    tm.assert_series_equal(result, s[0:2])
    result = s.iloc[0:1]
    tm.assert_series_equal(result, Series(dr[0:1]))
    result = pd.concat([s.iloc[0:1], s.iloc[1:]])
    tm.assert_series_equal(result, s)
    assert 'datetime64[ns, US/Eastern]' in str(s)
    result = s.shift()
    assert 'datetime64[ns, US/Eastern]' in str(result)
    assert 'NaT' in str(result)
    result = DatetimeIndex(s, freq='infer')
    tm.assert_index_equal(result, dr)