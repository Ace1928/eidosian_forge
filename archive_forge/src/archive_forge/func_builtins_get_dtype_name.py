import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def builtins_get_dtype_name(x):
    return _builtin_dtype_lookup[x.__class__]