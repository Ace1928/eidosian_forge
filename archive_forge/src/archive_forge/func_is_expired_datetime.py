from __future__ import annotations
import datetime
from lazyops.imports._dateparser import (
from typing import Optional, List, Union
def is_expired_datetime(dt: datetime.datetime, delta_days: Optional[int]=None, now: Optional[datetime.datetime]=None, tz: Optional[str]=None) -> bool:
    """
    Checks if the datetime is expired
    """
    if not now:
        now = get_current_datetime(tz=tz)
    if not delta_days:
        return now > dt
    dt = dt + datetime.timedelta(days=delta_days)
    return now > dt