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
def rrulely_breaks(info: date_breaks_info) -> Sequence[datetime]:
    r = rrule(info.frequency, interval=info.width, dtstart=info.start, until=info.until)
    return list(r)