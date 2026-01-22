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
def process_ellipses(obj, key, vdim_selection=False):
    """
    Helper function to pad a __getitem__ key with the right number of
    empty slices (i.e. :) when the key contains an Ellipsis (...).

    If the vdim_selection flag is true, check if the end of the key
    contains strings or Dimension objects in obj. If so, extra padding
    will not be applied for the value dimensions (i.e. the resulting key
    will be exactly one longer than the number of kdims). Note: this
    flag should not be used for composite types.
    """
    if getattr(getattr(key, 'dtype', None), 'kind', None) == 'b':
        return key
    wrapped_key = wrap_tuple(key)
    ellipse_count = sum((1 for k in wrapped_key if k is Ellipsis))
    if ellipse_count == 0:
        return key
    elif ellipse_count != 1:
        raise Exception('Only one ellipsis allowed at a time.')
    dim_count = len(obj.dimensions())
    index = wrapped_key.index(Ellipsis)
    head = wrapped_key[:index]
    tail = wrapped_key[index + 1:]
    padlen = dim_count - (len(head) + len(tail))
    if vdim_selection:
        if wrapped_key[-1] in obj.vdims:
            padlen = len(obj.kdims) + 1 - len(head + tail)
    return head + (slice(None),) * padlen + tail