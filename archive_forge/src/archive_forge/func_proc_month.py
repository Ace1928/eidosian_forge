from __future__ import absolute_import, print_function, division
import traceback as _traceback
import copy
import math
import re
import sys
import inspect
from time import time
import datetime
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzutc
import calendar
import binascii
import random
import pytz  # noqa
def proc_month(d):
    try:
        expanded[3].index('*')
    except ValueError:
        diff_month = nearest_diff_method(d.month, expanded[3], self.MONTHS_IN_YEAR)
        days = DAYS[month - 1]
        if month == 2 and self.is_leap(year) is True:
            days += 1
        reset_day = 1
        if diff_month is not None and diff_month != 0:
            if is_prev:
                d += relativedelta(months=diff_month)
                reset_day = DAYS[d.month - 1]
                if d.month == 2 and self.is_leap(d.year) is True:
                    reset_day += 1
                d += relativedelta(day=reset_day, hour=23, minute=59, second=59)
            else:
                d += relativedelta(months=diff_month, day=reset_day, hour=0, minute=0, second=0)
            return (True, d)
    return (False, d)