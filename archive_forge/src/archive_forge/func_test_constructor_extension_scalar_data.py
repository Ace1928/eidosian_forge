import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
@pytest.mark.parametrize('data,dtype', [(Period('2020-01'), PeriodDtype('M')), (Interval(left=0, right=5), IntervalDtype('int64', 'right')), (Timestamp('2011-01-01', tz='US/Eastern'), DatetimeTZDtype(unit='s', tz='US/Eastern'))])
def test_constructor_extension_scalar_data(self, data, dtype):
    df = DataFrame(index=[0, 1], columns=['a', 'b'], data=data)
    assert df['a'].dtype == dtype
    assert df['b'].dtype == dtype
    arr = pd.array([data] * 2, dtype=dtype)
    expected = DataFrame({'a': arr, 'b': arr})
    tm.assert_frame_equal(df, expected)