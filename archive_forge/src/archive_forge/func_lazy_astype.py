import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@lazy_cache('astype')
def lazy_astype(x, dtype_name):
    x = ensure_lazy(x)
    return x.to(fn=astype, args=(x, dtype_name))