from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def has_time(d: datetime) -> bool:
    """
    Return True if the time of datetime is not 00:00:00 (midnight)
    """
    return d.time() != time.min