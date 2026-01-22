from __future__ import annotations
import math
import typing
from datetime import datetime, timedelta, tzinfo
from typing import overload
from zoneinfo import ZoneInfo
import numpy as np
from dateutil.rrule import rrule
from ..utils import get_timezone, isclose_abs
from .date_utils import Interval, align_limits, expand_datetime_limits
from .types import DateFrequency, date_breaks_info
def datetime64_to_num(x: NDArrayDatetime) -> NDArrayFloat:
    """
    Convery any numpy datetime64 array to float array
    """
    x_secs = x.astype('datetime64[s]')
    diff_ns = (x - x_secs).astype('timedelta64[ns]')
    res = ((x_secs - EPOCH64).astype(np.float64) + diff_ns.astype(np.float64) / 1000000000.0) / SECONDS_PER_DAY
    x_int = x.astype(np.int64)
    res[x_int == NaT_int] = np.nan
    return res