import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def find_common_backend(*xs):
    names = tuple((x.backend if isinstance(x, LazyArray) else infer_backend(x) for x in xs))
    return _find_common_backend_cached(names)