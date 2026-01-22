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
@pytest.mark.parametrize('pydt', [True, False])
def test_constructor_data_aware_dtype_naive(self, tz_aware_fixture, pydt):
    tz = tz_aware_fixture
    ts = Timestamp('2019', tz=tz)
    if pydt:
        ts = ts.to_pydatetime()
    msg = 'Cannot convert timezone-aware data to timezone-naive dtype. Use pd.Series\\(values\\).dt.tz_localize\\(None\\) instead.'
    with pytest.raises(ValueError, match=msg):
        DataFrame({0: [ts]}, dtype='datetime64[ns]')
    msg2 = 'Cannot unbox tzaware Timestamp to tznaive dtype'
    with pytest.raises(TypeError, match=msg2):
        DataFrame({0: ts}, index=[0], dtype='datetime64[ns]')
    with pytest.raises(ValueError, match=msg):
        DataFrame([ts], dtype='datetime64[ns]')
    with pytest.raises(ValueError, match=msg):
        DataFrame(np.array([ts], dtype=object), dtype='datetime64[ns]')
    with pytest.raises(TypeError, match=msg2):
        DataFrame(ts, index=[0], columns=[0], dtype='datetime64[ns]')
    with pytest.raises(ValueError, match=msg):
        DataFrame([Series([ts])], dtype='datetime64[ns]')
    with pytest.raises(ValueError, match=msg):
        DataFrame([[ts]], columns=[0], dtype='datetime64[ns]')