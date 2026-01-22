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
def dimension_range(lower, upper, hard_range, soft_range, padding=None, log=False):
    """
    Computes the range along a dimension by combining the data range
    with the Dimension soft_range and range.
    """
    plower, pupper = range_pad(lower, upper, padding, log)
    if isfinite(soft_range[0]) and soft_range[0] <= lower:
        lower = soft_range[0]
    else:
        lower = max_range([(plower, None), (soft_range[0], None)])[0]
    if isfinite(soft_range[1]) and soft_range[1] >= upper:
        upper = soft_range[1]
    else:
        upper = max_range([(None, pupper), (None, soft_range[1])])[1]
    dmin, dmax = hard_range
    lower = lower if dmin is None or not isfinite(dmin) else dmin
    upper = upper if dmax is None or not isfinite(dmax) else dmax
    return (lower, upper)