from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('attr', ['values', 'asi8'])
@pytest.mark.parametrize('klass', [Index, DatetimeIndex])
def test_constructor_dtypes_datetime(self, tz_naive_fixture, attr, klass):
    index = date_range('2011-01-01', periods=5)
    arg = getattr(index, attr)
    index = index.tz_localize(tz_naive_fixture)
    dtype = index.dtype
    err = tz_naive_fixture is not None
    msg = 'Cannot use .astype to convert from timezone-naive dtype to'
    if attr == 'asi8':
        result = DatetimeIndex(arg).tz_localize(tz_naive_fixture)
        tm.assert_index_equal(result, index)
    elif klass is Index:
        with pytest.raises(TypeError, match='unexpected keyword'):
            klass(arg, tz=tz_naive_fixture)
    else:
        result = klass(arg, tz=tz_naive_fixture)
        tm.assert_index_equal(result, index)
    if attr == 'asi8':
        if err:
            with pytest.raises(TypeError, match=msg):
                DatetimeIndex(arg).astype(dtype)
        else:
            result = DatetimeIndex(arg).astype(dtype)
            tm.assert_index_equal(result, index)
    else:
        result = klass(arg, dtype=dtype)
        tm.assert_index_equal(result, index)
    if attr == 'asi8':
        result = DatetimeIndex(list(arg)).tz_localize(tz_naive_fixture)
        tm.assert_index_equal(result, index)
    elif klass is Index:
        with pytest.raises(TypeError, match='unexpected keyword'):
            klass(arg, tz=tz_naive_fixture)
    else:
        result = klass(list(arg), tz=tz_naive_fixture)
        tm.assert_index_equal(result, index)
    if attr == 'asi8':
        if err:
            with pytest.raises(TypeError, match=msg):
                DatetimeIndex(list(arg)).astype(dtype)
        else:
            result = DatetimeIndex(list(arg)).astype(dtype)
            tm.assert_index_equal(result, index)
    else:
        result = klass(list(arg), dtype=dtype)
        tm.assert_index_equal(result, index)