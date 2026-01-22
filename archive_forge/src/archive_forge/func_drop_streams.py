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
def drop_streams(streams, kdims, keys):
    """
    Drop any dimensioned streams from the keys and kdims.
    """
    stream_params = stream_parameters(streams)
    inds, dims = zip(*[(ind, kdim) for ind, kdim in enumerate(kdims) if kdim not in stream_params])
    get = operator.itemgetter(*inds)
    keys = (get(k) for k in keys)
    return (dims, [wrap_tuple(k) for k in keys] if len(inds) == 1 else list(keys))