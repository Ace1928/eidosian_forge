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
@pytest.mark.parametrize('fill_val', [1, 1.1, 1 + 1j, True, pd.Interval(1, 2, closed='left'), pd.Timestamp('2012-01-01', tz='US/Eastern'), pd.Timestamp('2012-01-01'), pd.Timedelta(days=1), pd.Period('2016-01-01', 'D')])
def test_fillna_interval(self, index_or_series, fill_val):
    ii = pd.interval_range(1.0, 5.0, closed='right').insert(1, np.nan)
    assert isinstance(ii.dtype, pd.IntervalDtype)
    obj = index_or_series(ii)
    exp = index_or_series([ii[0], fill_val, ii[2], ii[3], ii[4]], dtype=object)
    fill_dtype = object
    self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)