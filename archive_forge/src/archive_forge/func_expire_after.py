from __future__ import annotations
import calendar
import time
from datetime import datetime, timedelta, timezone
from email.utils import formatdate, parsedate, parsedate_tz
from typing import TYPE_CHECKING, Any, Mapping
def expire_after(delta: timedelta, date: datetime | None=None) -> datetime:
    date = date or datetime.now(timezone.utc)
    return date + delta