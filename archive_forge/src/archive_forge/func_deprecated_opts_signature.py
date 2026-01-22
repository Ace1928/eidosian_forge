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
def deprecated_opts_signature(args, kwargs):
    """
    Utility to help with the deprecation of the old .opts method signature

    Returns whether opts.apply_groups should be used (as a bool) and the
    corresponding options.
    """
    from .options import Options
    groups = set(Options._option_groups)
    opts = {kw for kw in kwargs if kw != 'clone'}
    apply_groups = False
    options = None
    new_kwargs = {}
    if len(args) > 0 and isinstance(args[0], dict):
        apply_groups = True
        if not set(args[0]).issubset(groups) and all((isinstance(v, dict) and (not set(v).issubset(groups) or not v) for v in args[0].values())):
            apply_groups = False
        elif set(args[0].keys()) <= groups:
            new_kwargs = args[0]
        else:
            options = args[0]
    elif opts and opts.issubset(set(groups)):
        apply_groups = True
    elif kwargs.get('options', None) is not None:
        apply_groups = True
    elif not args and (not kwargs):
        apply_groups = True
    return (apply_groups, options, new_kwargs)