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
def resolve_dependent_kwargs(kwargs):
    """Resolves parameter dependencies in the supplied dictionary

    Resolves parameter values, Parameterized instance methods and
    parameterized functions with dependencies in the supplied
    dictionary.

    Args:
       kwargs (dict): A dictionary of keyword arguments

    Returns:
       A new dictionary where any parameter dependencies have been
       resolved.
    """
    return {k: resolve_dependent_value(v) for k, v in kwargs.items()}