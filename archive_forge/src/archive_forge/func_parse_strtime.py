import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
def parse_strtime(timestr, fmt=PERFECT_TIME_FORMAT):
    """Turn a formatted time back into a datetime."""
    return datetime.datetime.strptime(timestr, fmt)