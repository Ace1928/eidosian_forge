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
def test_from_index(self):
    idx2 = date_range('20130101', periods=3, tz='US/Eastern', name='foo')
    df2 = DataFrame(idx2)
    tm.assert_series_equal(df2['foo'], Series(idx2, name='foo'))
    df2 = DataFrame(Series(idx2))
    tm.assert_series_equal(df2['foo'], Series(idx2, name='foo'))
    idx2 = date_range('20130101', periods=3, tz='US/Eastern')
    df2 = DataFrame(idx2)
    tm.assert_series_equal(df2[0], Series(idx2, name=0))
    df2 = DataFrame(Series(idx2))
    tm.assert_series_equal(df2[0], Series(idx2, name=0))