import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def datetime_mod(dt, period, start=None):
    """
    Find the time which is the specified date/time truncated to the time delta
    relative to the start date/time.
    By default, the start time is midnight of the same day as the specified
    date/time.

    >>> datetime_mod(datetime.datetime(2004, 1, 2, 3),
    ...     datetime.timedelta(days = 1.5),
    ...     start = datetime.datetime(2004, 1, 1))
    datetime.datetime(2004, 1, 1, 0, 0)
    >>> datetime_mod(datetime.datetime(2004, 1, 2, 13),
    ...     datetime.timedelta(days = 1.5),
    ...     start = datetime.datetime(2004, 1, 1))
    datetime.datetime(2004, 1, 2, 12, 0)
    >>> datetime_mod(datetime.datetime(2004, 1, 2, 13),
    ...     datetime.timedelta(days = 7),
    ...     start = datetime.datetime(2004, 1, 1))
    datetime.datetime(2004, 1, 1, 0, 0)
    >>> datetime_mod(datetime.datetime(2004, 1, 10, 13),
    ...     datetime.timedelta(days = 7),
    ...     start = datetime.datetime(2004, 1, 1))
    datetime.datetime(2004, 1, 8, 0, 0)
    """
    if start is None:
        start = datetime.datetime.combine(dt.date(), datetime.time())
    delta = dt - start

    def get_time_delta_microseconds(td):
        return (td.days * seconds_per_day + td.seconds) * 1000000 + td.microseconds
    delta, period = map(get_time_delta_microseconds, (delta, period))
    offset = datetime.timedelta(microseconds=delta % period)
    result = dt - offset
    return result