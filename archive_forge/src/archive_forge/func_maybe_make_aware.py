from __future__ import annotations
import numbers
import os
import random
import sys
import time as _time
from calendar import monthrange
from datetime import date, datetime, timedelta
from datetime import timezone as datetime_timezone
from datetime import tzinfo
from types import ModuleType
from typing import Any, Callable
from dateutil import tz as dateutil_tz
from dateutil.parser import isoparse
from kombu.utils.functional import reprcall
from kombu.utils.objects import cached_property
from .functional import dictfilter
from .text import pluralize
def maybe_make_aware(dt: datetime, tz: tzinfo | None=None, naive_as_utc: bool=True) -> datetime:
    """Convert dt to aware datetime, do nothing if dt is already aware."""
    if is_naive(dt):
        if naive_as_utc:
            dt = to_utc(dt)
        return localize(dt, timezone.utc if tz is None else timezone.tz_or_local(tz))
    return dt