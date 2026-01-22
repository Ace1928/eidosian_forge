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
@pytest.mark.parametrize('unit', ['D', 'h', 'm', 's', 'ms', 'us', 'ns'])
def test_constructor_timedelta_non_ns(self, order, unit):
    dtype = f'timedelta64[{unit}]'
    na = np.array([[np.timedelta64(1, 'D'), np.timedelta64(2, 'D')], [np.timedelta64(4, 'D'), np.timedelta64(5, 'D')]], dtype=dtype, order=order)
    df = DataFrame(na)
    if unit in ['D', 'h', 'm']:
        exp_unit = 's'
    else:
        exp_unit = unit
    exp_dtype = np.dtype(f'm8[{exp_unit}]')
    expected = DataFrame([[Timedelta(1, 'D'), Timedelta(2, 'D')], [Timedelta(4, 'D'), Timedelta(5, 'D')]], dtype=exp_dtype)
    tm.assert_frame_equal(df, expected)