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
def merge_options_to_dict(options):
    """
    Given a collection of Option objects or partial option dictionaries,
    merge everything to a single dictionary.
    """
    merged_options = {}
    for obj in options:
        if isinstance(obj, dict):
            new_opts = obj
        else:
            new_opts = {obj.key: obj.kwargs}
        merged_options = merge_option_dicts(merged_options, new_opts)
    return merged_options