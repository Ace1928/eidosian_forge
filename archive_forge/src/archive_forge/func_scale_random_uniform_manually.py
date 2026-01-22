import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def scale_random_uniform_manually(fn):

    @functools.wraps(fn)
    def numpy_like(low=0.0, high=1.0, size=None, dtype=None, **kwargs):
        if size is None:
            size = ()
        x = fn(size, **kwargs)
        if low != 0.0 or high != 1.0:
            x = (high - low) * x + low
        if dtype is not None and get_dtype_name(x) != dtype:
            x = astype(x, dtype)
        return x
    return numpy_like