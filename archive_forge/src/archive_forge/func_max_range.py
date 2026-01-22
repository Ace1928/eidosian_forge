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
def max_range(ranges, combined=True):
    """
    Computes the maximal lower and upper bounds from a list bounds.

    Args:
       ranges (list of tuples): A list of range tuples
       combined (boolean, optional): Whether to combine bounds
          Whether range should be computed on lower and upper bound
          independently or both at once

    Returns:
       The maximum range as a single tuple
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
            values = [tuple((np.nan if v is None else v for v in r)) for r in ranges]
            if any((isinstance(v, datetime_types) and (not isinstance(v, cftime_types + (dt.time,))) for r in values for v in r)):
                converted = []
                for l, h in values:
                    if isinstance(l, pd.Period) and isinstance(h, pd.Period):
                        l = l.to_timestamp().to_datetime64()
                        h = h.to_timestamp().to_datetime64()
                    elif isinstance(l, datetime_types) and isinstance(h, datetime_types):
                        l, h = (pd.Timestamp(l).to_datetime64(), pd.Timestamp(h).to_datetime64())
                    converted.append((l, h))
                values = converted
            arr = np.array(values)
            if not len(arr):
                return (np.nan, np.nan)
            elif arr.dtype.kind in 'OSU':
                arr = list(python2sort([v for r in values for v in r if not is_nan(v) and v is not None]))
                return (arr[0], arr[-1])
            elif arr.dtype.kind in 'M':
                drange = (arr.min(), arr.max()) if combined else (arr[:, 0].min(), arr[:, 1].max())
                return drange
            if combined:
                return (np.nanmin(arr), np.nanmax(arr))
            else:
                return (np.nanmin(arr[:, 0]), np.nanmax(arr[:, 1]))
    except Exception:
        return (np.nan, np.nan)