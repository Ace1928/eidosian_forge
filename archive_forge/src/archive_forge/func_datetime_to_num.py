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
def datetime_to_num(x: SeqDatetime | Datetime) -> NDArrayFloat | float:
    """
    Convery any datetime sequence to float array
    """
    iterable = np.iterable(x)
    _x = x if iterable else [x]
    try:
        x0 = next(iter(_x))
    except StopIteration:
        return np.array([], dtype=float)
    if isinstance(x0, datetime) and x0.tzinfo:
        _x = [dt.astimezone(UTC).replace(tzinfo=None) for dt in _x]
    res = datetime64_to_num(np.asarray(_x, dtype='datetime64'))
    return res if iterable else res[0]