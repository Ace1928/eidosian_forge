import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
def is_soon(dt, window):
    """Determines if time is going to happen in the next window seconds.

    :param dt: the time
    :param window: minimum seconds to remain to consider the time not soon

    :return: True if expiration is within the given duration
    """
    soon = utcnow() + datetime.timedelta(seconds=window)
    return normalize_time(dt) <= soon