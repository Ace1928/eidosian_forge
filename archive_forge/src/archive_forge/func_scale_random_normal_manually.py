import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def scale_random_normal_manually(fn):

    @functools.wraps(fn)
    def numpy_like(loc=0.0, scale=1.0, size=None, dtype=None, **kwargs):
        if size is None:
            size = ()
        x = fn(size=size, **kwargs)
        if loc != 0.0 or scale != 1.0:
            x = scale * x + loc
        if dtype is not None and get_dtype_name(x) != dtype:
            x = astype(x, dtype)
        return x
    return numpy_like