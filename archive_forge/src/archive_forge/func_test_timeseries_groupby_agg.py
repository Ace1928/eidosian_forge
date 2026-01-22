import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_timeseries_groupby_agg():

    def func(ser):
        if ser.isna().all():
            return None
        return np.sum(ser)
    df = DataFrame([1.0], index=[pd.Timestamp('2018-01-16 00:00:00+00:00')])
    res = df.groupby(lambda x: 1).agg(func)
    expected = DataFrame([[1.0]], index=[1])
    tm.assert_frame_equal(res, expected)