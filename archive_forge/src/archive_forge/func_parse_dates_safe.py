from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
def parse_dates_safe(dates: Series, delta: bool=False, year: bool=False, days: bool=False):
    d = {}
    if lib.is_np_dtype(dates.dtype, 'M'):
        if delta:
            time_delta = dates - Timestamp(stata_epoch).as_unit('ns')
            d['delta'] = time_delta._values.view(np.int64) // 1000
        if days or year:
            date_index = DatetimeIndex(dates)
            d['year'] = date_index._data.year
            d['month'] = date_index._data.month
        if days:
            days_in_ns = dates._values.view(np.int64) - to_datetime(d['year'], format='%Y')._values.view(np.int64)
            d['days'] = days_in_ns // NS_PER_DAY
    elif infer_dtype(dates, skipna=False) == 'datetime':
        if delta:
            delta = dates._values - stata_epoch

            def f(x: timedelta) -> float:
                return US_PER_DAY * x.days + 1000000 * x.seconds + x.microseconds
            v = np.vectorize(f)
            d['delta'] = v(delta)
        if year:
            year_month = dates.apply(lambda x: 100 * x.year + x.month)
            d['year'] = year_month._values // 100
            d['month'] = year_month._values - d['year'] * 100
        if days:

            def g(x: datetime) -> int:
                return (x - datetime(x.year, 1, 1)).days
            v = np.vectorize(g)
            d['days'] = v(dates)
    else:
        raise ValueError('Columns containing dates must contain either datetime64, datetime or null values.')
    return DataFrame(d, index=index)