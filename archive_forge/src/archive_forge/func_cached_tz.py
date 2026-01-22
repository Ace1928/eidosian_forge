from __future__ import annotations
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from functools import lru_cache
import re
from typing import Any
from ._types import ParseFloat
@lru_cache(maxsize=None)
def cached_tz(hour_str: str, minute_str: str, sign_str: str) -> timezone:
    sign = 1 if sign_str == '+' else -1
    return timezone(timedelta(hours=sign * int(hour_str), minutes=sign * int(minute_str)))