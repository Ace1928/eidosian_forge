import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_is_datetimelike_array_all_nan_nat_like(self):
    arr = np.array([np.nan, pd.NaT, np.datetime64('nat')])
    assert lib.is_datetime_array(arr)
    assert lib.is_datetime64_array(arr)
    assert not lib.is_timedelta_or_timedelta64_array(arr)
    arr = np.array([np.nan, pd.NaT, np.timedelta64('nat')])
    assert not lib.is_datetime_array(arr)
    assert not lib.is_datetime64_array(arr)
    assert lib.is_timedelta_or_timedelta64_array(arr)
    arr = np.array([np.nan, pd.NaT, np.datetime64('nat'), np.timedelta64('nat')])
    assert not lib.is_datetime_array(arr)
    assert not lib.is_datetime64_array(arr)
    assert not lib.is_timedelta_or_timedelta64_array(arr)
    arr = np.array([np.nan, pd.NaT])
    assert lib.is_datetime_array(arr)
    assert lib.is_datetime64_array(arr)
    assert lib.is_timedelta_or_timedelta64_array(arr)
    arr = np.array([np.nan, np.nan], dtype=object)
    assert not lib.is_datetime_array(arr)
    assert not lib.is_datetime64_array(arr)
    assert not lib.is_timedelta_or_timedelta64_array(arr)
    assert lib.is_datetime_with_singletz_array(np.array([Timestamp('20130101', tz='US/Eastern'), Timestamp('20130102', tz='US/Eastern')], dtype=object))
    assert not lib.is_datetime_with_singletz_array(np.array([Timestamp('20130101', tz='US/Eastern'), Timestamp('20130102', tz='CET')], dtype=object))