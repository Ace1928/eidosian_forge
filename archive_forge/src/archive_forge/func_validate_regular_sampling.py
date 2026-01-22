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
def validate_regular_sampling(values, rtol=1e-05):
    """
    Validates regular sampling of a 1D array ensuring that the difference
    in sampling steps is at most rtol times the smallest sampling step.
    Returns a boolean indicating whether the sampling is regular.
    """
    diffs = np.diff(values)
    return len(diffs) < 1 or abs(diffs.min() - diffs.max()) < abs(diffs.min() * rtol)