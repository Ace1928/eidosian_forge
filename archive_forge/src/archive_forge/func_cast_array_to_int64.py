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
def cast_array_to_int64(array):
    """
    Convert a numpy array  to `int64`. Suppress the following warning
    emitted by Numpy, which as of 12/2021 has been extensively discussed
    (https://github.com/pandas-dev/pandas/issues/22384)
    and whose fate (possible revert) has not yet been settled:

        FutureWarning: casting datetime64[ns] values to int64 with .astype(...)
        is deprecated and will raise in a future version. Use .view(...) instead.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='casting datetime64', category=FutureWarning)
        return array.astype('int64')