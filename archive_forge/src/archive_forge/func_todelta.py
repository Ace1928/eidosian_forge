from abc import abstractmethod
import math
import operator
import re
import datetime
from calendar import isleap
from decimal import Decimal, Context
from typing import cast, Any, Callable, Dict, Optional, Tuple, Union
from ..helpers import MONTH_DAYS_LEAP, MONTH_DAYS, DAYS_IN_4Y, \
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
def todelta(self) -> datetime.timedelta:
    """Returns the datetime.timedelta from 0001-01-01T00:00:00 CE."""
    if self._year is None:
        delta = operator.sub(*self._get_operands(datetime.datetime(1, 1, 1)))
        return cast(datetime.timedelta, delta)
    year, dt = (self.year, self._dt)
    tzinfo = None if dt.tzinfo is None else self._utc_timezone
    if year > 0:
        m_days = MONTH_DAYS_LEAP if isleap(year) else MONTH_DAYS
        days = days_from_common_era(year - 1) + sum((m_days[m] for m in range(1, dt.month)))
    else:
        m_days = MONTH_DAYS_LEAP if isleap(year + 1) else MONTH_DAYS
        days = days_from_common_era(year) + sum((m_days[m] for m in range(1, dt.month)))
    delta = dt - datetime.datetime(dt.year, dt.month, day=1, tzinfo=tzinfo)
    return datetime.timedelta(days=days, seconds=delta.total_seconds())