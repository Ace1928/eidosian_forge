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
def get_overlay_spec(o, k, v):
    """
    Gets the type.group.label + key spec from an Element in an Overlay.
    """
    k = wrap_tuple(k)
    return (type(v).__name__, v.group, v.label) + k if len(o.kdims) else (type(v).__name__,) + k