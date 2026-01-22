import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_basic_dtype(self):
    msg = 'is_interval_dtype is deprecated'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert is_interval_dtype('interval[int64, both]')
        assert is_interval_dtype(IntervalIndex.from_tuples([(0, 1)]))
        assert is_interval_dtype(IntervalIndex.from_breaks(np.arange(4)))
        assert is_interval_dtype(IntervalIndex.from_breaks(date_range('20130101', periods=3)))
        assert not is_interval_dtype('U')
        assert not is_interval_dtype('S')
        assert not is_interval_dtype('foo')
        assert not is_interval_dtype(np.object_)
        assert not is_interval_dtype(np.int64)
        assert not is_interval_dtype(np.float64)