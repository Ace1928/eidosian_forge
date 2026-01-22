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
def microsecondly_breaks(info: date_breaks_info) -> NDArrayDatetime:
    """
    Calculate breaks at microsecond intervals
    """
    nmin: float
    nmax: float
    width = info.width
    nmin, nmax = datetime_to_num((info.start, info.until))
    day0: float = np.floor(nmin)
    umax = (nmax - day0) * MICROSECONDS_PER_DAY
    umin = (nmin - day0) * MICROSECONDS_PER_DAY
    width = info.width
    h, m = divmod(umax, width)
    if not isclose_abs(m / width, 0):
        h += 1
    umax = h * width
    n = (umax - umin + 0.001 * width) // width
    ubreaks = umin - width + np.arange(n + 3) * width
    breaks = day0 + ubreaks / MICROSECONDS_PER_DAY
    return num_to_datetime(breaks, info.tz)