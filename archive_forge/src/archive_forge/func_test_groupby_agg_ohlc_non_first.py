from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_agg_ohlc_non_first():
    df = DataFrame([[1], [1]], columns=Index(['foo'], name='mycols'), index=date_range('2018-01-01', periods=2, freq='D', name='dti'))
    expected = DataFrame([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], columns=MultiIndex.from_tuples((('foo', 'sum', 'foo'), ('foo', 'ohlc', 'open'), ('foo', 'ohlc', 'high'), ('foo', 'ohlc', 'low'), ('foo', 'ohlc', 'close')), names=['mycols', None, None]), index=date_range('2018-01-01', periods=2, freq='D', name='dti'))
    result = df.groupby(Grouper(freq='D')).agg(['sum', 'ohlc'])
    tm.assert_frame_equal(result, expected)