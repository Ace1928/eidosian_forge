from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_index_reset_index_dt64tz(self):
    idx = Index(date_range('20130101', periods=3, tz='US/Eastern'), name='foo')
    df = DataFrame({'A': [0, 1, 2]}, index=idx)
    result = df.reset_index()
    assert result['foo'].dtype == 'datetime64[ns, US/Eastern]'
    df = result.set_index('foo')
    tm.assert_index_equal(df.index, idx)