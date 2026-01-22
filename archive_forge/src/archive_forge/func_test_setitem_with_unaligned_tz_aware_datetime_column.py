from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_with_unaligned_tz_aware_datetime_column(self):
    column = Series(date_range('2015-01-01', periods=3, tz='utc'), name='dates')
    df = DataFrame({'dates': column})
    df['dates'] = column[[1, 0, 2]]
    tm.assert_series_equal(df['dates'], column)
    df = DataFrame({'dates': column})
    df.loc[[0, 1, 2], 'dates'] = column[[1, 0, 2]]
    tm.assert_series_equal(df['dates'], column)