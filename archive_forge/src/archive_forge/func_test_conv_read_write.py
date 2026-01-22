import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_conv_read_write():
    with tm.ensure_clean() as path:

        def roundtrip(key, obj, **kwargs):
            obj.to_hdf(path, key=key, **kwargs)
            return read_hdf(path, key)
        o = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
        tm.assert_series_equal(o, roundtrip('series', o))
        o = Series(range(10), dtype='float64', index=[f'i_{i}' for i in range(10)])
        tm.assert_series_equal(o, roundtrip('string_series', o))
        o = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
        tm.assert_frame_equal(o, roundtrip('frame', o))
        df = DataFrame({'A': range(5), 'B': range(5)})
        df.to_hdf(path, key='table', append=True)
        result = read_hdf(path, 'table', where=['index>2'])
        tm.assert_frame_equal(df[df.index > 2], result)