from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def floor_second(d: datetime) -> datetime:
    """
    Round down to the start of the second
    """
    if at_the_second(d):
        return d
    return floor_minute(d).replace(second=d.second)