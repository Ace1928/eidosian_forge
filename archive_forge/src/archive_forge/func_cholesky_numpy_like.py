import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
@functools.wraps(fn)
def cholesky_numpy_like(a):
    return fn(a, lower=True)