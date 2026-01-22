import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
def is_newer_than(after, seconds):
    """Return True if after is newer than seconds.

    .. versionchanged:: 1.7
       Accept datetime string with timezone information.
       Fix comparison with timezone aware datetime.
    """
    if isinstance(after, str):
        after = parse_isotime(after)
    after = normalize_time(after)
    return after - utcnow() > datetime.timedelta(seconds=seconds)