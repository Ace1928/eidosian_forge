from __future__ import annotations
import re
from bisect import bisect, bisect_left
from collections import namedtuple
from collections.abc import Iterable
from datetime import datetime, timedelta, tzinfo
from typing import Any, Callable, Mapping, Sequence
from kombu.utils.objects import cached_property
from celery import Celery
from . import current_app
from .utils.collections import AttributeDict
from .utils.time import (ffwd, humanize_seconds, localize, maybe_make_aware, maybe_timedelta, remaining, timezone,
def is_due(self, last_run_at: datetime) -> tuple[bool, datetime]:
    """Return tuple of ``(is_due, next_time_to_run)``.

        Note:
            next time to run is in seconds.

        See Also:
            :meth:`celery.schedules.schedule.is_due` for more information.
        """
    rem_delta = self.remaining_estimate(last_run_at)
    rem = max(rem_delta.total_seconds(), 0)
    due = rem == 0
    if due:
        rem_delta = self.remaining_estimate(self.now())
        rem = max(rem_delta.total_seconds(), 0)
    return schedstate(due, rem)