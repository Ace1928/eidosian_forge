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
def deephash(obj):
    """
    Given an object, return a hash using HashableJSON. This hash is not
    architecture, Python version or platform independent.
    """
    try:
        return hash(json.dumps(obj, cls=HashableJSON, sort_keys=True))
    except Exception:
        return None