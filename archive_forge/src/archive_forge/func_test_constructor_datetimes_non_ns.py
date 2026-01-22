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
@pytest.mark.parametrize('order', ['K', 'A', 'C', 'F'])
@pytest.mark.parametrize('unit', ['M', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'])
def test_constructor_datetimes_non_ns(self, order, unit):
    dtype = f'datetime64[{unit}]'
    na = np.array([['2015-01-01', '2015-01-02', '2015-01-03'], ['2017-01-01', '2017-01-02', '2017-02-03']], dtype=dtype, order=order)
    df = DataFrame(na)
    expected = DataFrame(na.astype('M8[ns]'))
    if unit in ['M', 'D', 'h', 'm']:
        with pytest.raises(TypeError, match='Cannot cast'):
            expected.astype(dtype)
        expected = expected.astype('datetime64[s]')
    else:
        expected = expected.astype(dtype=dtype)
    tm.assert_frame_equal(df, expected)