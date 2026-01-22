import calendar
from datetime import datetime
from datetime import timedelta
import re
import sys
import time
def time_a_while_ago(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0):
    """
    format of timedelta:
        timedelta([days[, seconds[, microseconds[, milliseconds[,
                    minutes[, hours[, weeks]]]]]]])
    """
    delta = timedelta(days, seconds, microseconds, milliseconds, minutes, hours, weeks)
    return datetime.utcnow() - delta