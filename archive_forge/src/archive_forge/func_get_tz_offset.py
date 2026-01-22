import calendar
import datetime
import logging
import os
import time
import warnings
from tzlocal import windows_tz
def get_tz_offset(tz):
    """Get timezone's offset using built-in function datetime.utcoffset()."""
    return int(datetime.datetime.now(tz).utcoffset().total_seconds())