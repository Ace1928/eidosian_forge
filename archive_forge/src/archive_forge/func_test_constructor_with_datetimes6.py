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
def test_constructor_with_datetimes6(self):
    i = date_range('1/1/2011', periods=5, freq='10s', tz='US/Eastern')
    i_no_tz = date_range('1/1/2011', periods=5, freq='10s')
    df = DataFrame({'a': i, 'b': i_no_tz})
    expected = DataFrame({'a': i.to_series().reset_index(drop=True), 'b': i_no_tz})
    tm.assert_frame_equal(df, expected)