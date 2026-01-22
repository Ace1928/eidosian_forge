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
def proc_day_of_week(d):
    try:
        expanded[4].index('*')
    except ValueError:
        diff_day_of_week = nearest_diff_method(d.isoweekday() % 7, expanded[4], 7)
        if diff_day_of_week is not None and diff_day_of_week != 0:
            if is_prev:
                d += relativedelta(days=diff_day_of_week, hour=23, minute=59, second=59)
            else:
                d += relativedelta(days=diff_day_of_week, hour=0, minute=0, second=0)
            return (True, d)
    return (False, d)