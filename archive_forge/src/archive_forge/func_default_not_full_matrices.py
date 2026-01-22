import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
@functools.wraps(fn)
def default_not_full_matrices(*args, **kwargs):
    kwargs.setdefault('full_matrices', False)
    return fn(*args, **kwargs)