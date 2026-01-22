import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def with_dtype_wrapper(fn):
    """Add ability to handle `dtype` keyword.
    If not None, `dtype` should be specified as a string, otherwise conversion
    will happen regardless.
    """

    @functools.wraps(fn)
    def with_dtype(*args, dtype=None, **kwargs):
        A = fn(*args, **kwargs)
        if dtype is not None and dtype != get_dtype_name(A):
            A = astype(A, dtype)
        return A
    return with_dtype