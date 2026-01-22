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
@pytest.mark.parametrize('extension_arr', [Categorical(list('aabbc')), SparseArray([1, np.nan, np.nan, np.nan]), IntervalArray([Interval(0, 1), Interval(1, 5)]), PeriodArray(pd.period_range(start='1/1/2017', end='1/1/2018', freq='M'))])
def test_constructor_with_extension_array(self, extension_arr):
    expected = DataFrame(Series(extension_arr))
    result = DataFrame(extension_arr)
    tm.assert_frame_equal(result, expected)