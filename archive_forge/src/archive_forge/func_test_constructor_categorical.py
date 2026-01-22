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
def test_constructor_categorical(self):
    df = DataFrame({'A': list('abc')}, dtype='category')
    expected = Series(list('abc'), dtype='category', name='A')
    tm.assert_series_equal(df['A'], expected)
    s = Series(list('abc'), dtype='category')
    result = s.to_frame()
    expected = Series(list('abc'), dtype='category', name=0)
    tm.assert_series_equal(result[0], expected)
    result = s.to_frame(name='foo')
    expected = Series(list('abc'), dtype='category', name='foo')
    tm.assert_series_equal(result['foo'], expected)
    df = DataFrame(list('abc'), dtype='category')
    expected = Series(list('abc'), dtype='category', name=0)
    tm.assert_series_equal(df[0], expected)