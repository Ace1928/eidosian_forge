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
def get_tzinfo(tz: Optional[str | TzInfo]=None) -> TzInfo | None:
    """
    Generate `~datetime.tzinfo` from a string or return `~datetime.tzinfo`.

    If argument is None, return None.
    """
    if tz is None:
        return None
    if isinstance(tz, str):
        return ZoneInfo(tz)
    if isinstance(tz, tzinfo):
        return tz
    raise TypeError('tz must be string or tzinfo subclass.')