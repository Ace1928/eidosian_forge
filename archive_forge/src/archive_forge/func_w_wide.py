from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
@property
def w_wide(self) -> int:
    """
        Weeks (enclosing the original)
        """
    return Interval(*self.limits_week()).w