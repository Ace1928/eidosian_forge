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
def test_constructor_dict_order_insertion(self):
    datetime_series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
    datetime_series_short = datetime_series[:5]
    d = {'b': datetime_series_short, 'a': datetime_series}
    frame = DataFrame(data=d)
    expected = DataFrame(data=d, columns=list('ba'))
    tm.assert_frame_equal(frame, expected)