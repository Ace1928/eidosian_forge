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
def get_exponential_backoff_interval(factor: int, retries: int, maximum: int, full_jitter: bool=False) -> int:
    """Calculate the exponential backoff wait time."""
    countdown = min(maximum, factor * 2 ** retries)
    if full_jitter:
        countdown = random.randrange(countdown + 1)
    return max(0, countdown)