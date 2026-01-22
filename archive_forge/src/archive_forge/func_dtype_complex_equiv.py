import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@functools.lru_cache(None)
def dtype_complex_equiv(dtype_name):
    return _DTYPES_COMPLEX_EQUIV.get(dtype_name, dtype_name)