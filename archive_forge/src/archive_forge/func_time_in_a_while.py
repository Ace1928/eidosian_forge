import calendar
from datetime import datetime
from datetime import timedelta
import re
import sys
import time
def time_in_a_while(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0):
    """
    format of timedelta:
        timedelta([days[, seconds[, microseconds[, milliseconds[,
                    minutes[, hours[, weeks]]]]]]])
    :return: UTC time
    """
    delta = timedelta(days, seconds, microseconds, milliseconds, minutes, hours, weeks)
    return datetime.utcnow() + delta