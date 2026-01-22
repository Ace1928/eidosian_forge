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
def parse_datetime_selection(sel):
    """
    Parses string selection specs as datetimes.
    """
    if isinstance(sel, str) or isdatetime(sel):
        sel = parse_datetime(sel)
    if isinstance(sel, slice):
        if isinstance(sel.start, str) or isdatetime(sel.start):
            sel = slice(parse_datetime(sel.start), sel.stop)
        if isinstance(sel.stop, str) or isdatetime(sel.stop):
            sel = slice(sel.start, parse_datetime(sel.stop))
    if isinstance(sel, (set, list)):
        sel = [parse_datetime(v) if isinstance(v, str) else v for v in sel]
    return sel