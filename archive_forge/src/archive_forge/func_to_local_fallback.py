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
def to_local_fallback(self, dt: datetime) -> datetime:
    """Converts a datetime to the local timezone, or the system timezone."""
    if is_naive(dt):
        return make_aware(dt, self.local)
    return localize(dt, self.local)