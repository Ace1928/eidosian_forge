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
def range_pad(lower, upper, padding=None, log=False):
    """
    Pads the range by a fraction of the interval
    """
    if padding is not None and (not isinstance(padding, tuple)):
        padding = (padding, padding)
    if is_number(lower) and is_number(upper) and (padding is not None):
        if not isinstance(lower, datetime_types) and log and (lower > 0) and (upper > 0):
            log_min = np.log(lower) / np.log(10)
            log_max = np.log(upper) / np.log(10)
            lspan = (log_max - log_min) * (1 + padding[0] * 2)
            uspan = (log_max - log_min) * (1 + padding[1] * 2)
            center = (log_min + log_max) / 2.0
            start, end = (np.power(10, center - lspan / 2.0), np.power(10, center + uspan / 2.0))
        else:
            if isinstance(lower, datetime_types) and (not isinstance(lower, cftime_types)):
                lower, upper = (np.datetime64(lower), np.datetime64(upper))
                span = (upper - lower).astype('>m8[ns]')
            else:
                span = upper - lower
            lpad = span * padding[0]
            upad = span * padding[1]
            start, end = (lower - lpad, upper + upad)
    else:
        start, end = (lower, upper)
    return (start, end)