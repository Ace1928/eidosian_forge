from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def limits_year(self) -> TupleDatetime2:
    return (floor_year(self.start), ceil_year(self.end))