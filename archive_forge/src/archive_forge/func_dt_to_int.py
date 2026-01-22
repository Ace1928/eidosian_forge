import builtins
import datetime as dt
import hashlib
import inspect
import itertools
import json
import numbers
import operator
import pickle
import string
import sys
import time
import types
import unicodedata
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import partial
from threading import Event, Thread
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
def dt_to_int(value, time_unit='us'):
    """
    Converts a datetime type to an integer with the supplied time unit.
    """
    if isinstance(value, pd.Period):
        value = value.to_timestamp()
    if isinstance(value, pd.Timestamp):
        try:
            value = value.to_datetime64()
        except Exception:
            value = np.datetime64(value.to_pydatetime())
    if isinstance(value, cftime_types):
        return cftime_to_timestamp(value, time_unit)
    if isinstance(value, dt.date) and (not isinstance(value, dt.datetime)):
        value = dt.datetime(*value.timetuple()[:6])
    if isinstance(value, np.datetime64):
        try:
            value = np.datetime64(value, 'ns')
            tscale = np.timedelta64(1, time_unit) / np.timedelta64(1, 'ns')
            return int(value.tolist() / tscale)
        except Exception:
            value = value.tolist()
    if time_unit == 'ns':
        tscale = 1000000000.0
    else:
        tscale = 1.0 / np.timedelta64(1, time_unit).tolist().total_seconds()
    if value.tzinfo is None:
        _epoch = dt.datetime(1970, 1, 1)
    else:
        _epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    return int((value - _epoch).total_seconds() * tscale)