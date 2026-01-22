import datetime
import decimal
import functools
import logging
import time
import warnings
from contextlib import contextmanager
from hashlib import md5
from django.apps import apps
from django.db import NotSupportedError
from django.utils.dateparse import parse_time
def typecast_timestamp(s):
    if not s:
        return None
    if ' ' not in s:
        return typecast_date(s)
    d, t = s.split()
    if '-' in t:
        t, _ = t.split('-', 1)
    elif '+' in t:
        t, _ = t.split('+', 1)
    dates = d.split('-')
    times = t.split(':')
    seconds = times[2]
    if '.' in seconds:
        seconds, microseconds = seconds.split('.')
    else:
        microseconds = '0'
    return datetime.datetime(int(dates[0]), int(dates[1]), int(dates[2]), int(times[0]), int(times[1]), int(seconds), int((microseconds + '000000')[:6]))