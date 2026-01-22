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
def tree_attribute(identifier):
    """
    Predicate that returns True for custom attributes added to AttrTrees
    that are not methods, properties or internal attributes.

    These custom attributes start with a capitalized character when
    applicable (not applicable to underscore or certain unicode characters)
    """
    if identifier == '':
        return True
    if identifier[0].upper().isupper() is False and identifier[0] != '_':
        return True
    else:
        return identifier[0].isupper()