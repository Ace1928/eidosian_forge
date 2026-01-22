import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_str(self):
    a = Series(date_range('2010-01-04', periods=5))
    b = Series(date_range('3/6/2012 00:00', periods=5, tz='US/Eastern'))
    c = Series([Timedelta(x, unit='d') for x in range(5)])
    d = Series(range(5))
    e = Series([0.0, 0.2, 0.4, 0.6, 0.8])
    df = DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
    result = df.astype(str)
    expected = DataFrame({'a': list(map(str, (Timestamp(x)._date_repr for x in a._values))), 'b': list(map(str, map(Timestamp, b._values))), 'c': [Timedelta(x)._repr_base() for x in c._values], 'd': list(map(str, d._values)), 'e': list(map(str, e._values))}, dtype='object')
    tm.assert_frame_equal(result, expected)