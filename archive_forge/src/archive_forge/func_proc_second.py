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
def proc_second(d):
    if len(expanded) == 6:
        try:
            expanded[5].index('*')
        except ValueError:
            diff_sec = nearest_diff_method(d.second, expanded[5], 60)
            if diff_sec is not None and diff_sec != 0:
                d += relativedelta(seconds=diff_sec)
                return (True, d)
    else:
        d += relativedelta(second=0)
    return (False, d)