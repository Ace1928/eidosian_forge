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
def roll_over() -> None:
    for _ in range(2000):
        flag = datedata.dom == len(days_of_month) or day_out_of_range(datedata.year, months_of_year[datedata.moy], days_of_month[datedata.dom]) or is_before_last_run(datedata.year, months_of_year[datedata.moy], days_of_month[datedata.dom])
        if flag:
            datedata.dom = 0
            datedata.moy += 1
            if datedata.moy == len(months_of_year):
                datedata.moy = 0
                datedata.year += 1
        else:
            break
    else:
        raise RuntimeError('unable to rollover, time specification is probably invalid')