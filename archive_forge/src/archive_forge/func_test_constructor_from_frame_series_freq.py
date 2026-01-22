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
def test_constructor_from_frame_series_freq(self, using_infer_string):
    dts = ['1-1-1990', '2-1-1990', '3-1-1990', '4-1-1990', '5-1-1990']
    expected = DatetimeIndex(dts, freq='MS')
    df = DataFrame(np.random.default_rng(2).random((5, 3)))
    df['date'] = dts
    result = DatetimeIndex(df['date'], freq='MS')
    dtype = object if not using_infer_string else 'string'
    assert df['date'].dtype == dtype
    expected.name = 'date'
    tm.assert_index_equal(result, expected)
    expected = Series(dts, name='date')
    tm.assert_series_equal(df['date'], expected)
    if not using_infer_string:
        freq = pd.infer_freq(df['date'])
        assert freq == 'MS'