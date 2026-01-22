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
def calculate_date_breaks_info(limits: tuple[datetime, datetime], n: int=5) -> date_breaks_info:
    """
    Calculate information required to generate breaks

    Parameters
    ----------
    limits:
        Datetime limits for the breaks
    n:
        Desired number of breaks.
    """
    tz = get_timezone(limits)
    _max_breaks_lookup = {f: max(b, n) for f, b in MAX_BREAKS.items()}
    itv = Interval(*limits)
    unit_durations = (itv.y_wide, itv.M_wide, itv.d_wide, itv.h_wide, itv.m_wide, itv.s, itv.u)
    freq = DF.YEARLY
    break_width = 1
    duration = n
    for freq, duration in _viable_freqs(n, unit_durations):
        _max_breaks = _max_breaks_lookup[freq]
        for mb in range(n, _max_breaks + 1):
            if duration < mb:
                continue
            for break_width in WIDTHS[freq]:
                if duration <= break_width * mb - 1:
                    break
            else:
                continue
            break
        else:
            continue
        break
    num_breaks = duration // break_width
    limits = itv.limits_for_frequency(freq)
    res = date_breaks_info(freq, num_breaks, break_width, start=limits[0].replace(tzinfo=tz), until=limits[1].replace(tzinfo=tz), tz=tz)
    return res