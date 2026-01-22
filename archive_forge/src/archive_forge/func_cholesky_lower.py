import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def cholesky_lower(fn):

    @functools.wraps(fn)
    def cholesky_numpy_like(a):
        return fn(a, lower=True)
    return cholesky_numpy_like