import time
import logging
import datetime
import functools
from pyzor.engines.common import *
def encode_date(date):
    """Convert the date to Unix Timestamp"""
    if date is None:
        return 0
    return int(time.mktime(date.timetuple()))